"""Stage 2: open/closed-vocab recognition + clean masks via SAM 2.

We offer three interchangeable *detector* backends, all paired with SAM 2 for
segmentation so the rest of the pipeline (mask cleanup, depth band, parallax
warp, tracker) is detector-agnostic:

  - DinoRecognizer        — Grounding DINO Tiny (open-vocab text prompt)
                            via transformers; the original backend.
  - YoloRecognizer        — Ultralytics YOLO11n (closed-vocab, COCO 80) —
                            very fast, no posters / bodhran / drum-frame.
  - MediapipeRecognizer   — MediaPipe Object Detector EfficientDet-Lite0
                            (closed-vocab, COCO 80) — fast on CPU, same
                            blind spots as YOLO.

All three implement the same public surface used by `pipeline.Pipeline`:

    label_image(color_bgr, prompt=..., box_threshold=..., text_threshold=...)
        -> List[{"label": str, "score": float, "bbox": (x, y, w, h)}]
    segment_with_points(color_bgr, points)
        -> List[(mask_uint8, score)]   # uint8 {0,255}, camera-space

`prompt`, `box_threshold`, `text_threshold` are honored by DINO and ignored
(silently) by YOLO/MediaPipe.

Pick a backend with:

    rec = create_recognizer("dino" | "yolo" | "mediapipe")

The active backend is persisted in `runtime/active_detector.json` and read
by the perception daemon at startup. The web UI's detector dropdown writes
that file and then restarts the perception task.

Heavy imports (torch, transformers, ultralytics, mediapipe) are deferred
to first instantiation so importing this module is cheap.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .types import Blob


DEFAULT_DINO_PROMPT = (
    "guitar. acoustic guitar. electric guitar. bass guitar. ukulele. "
    "drum. drum head. snare drum. bodhran. frame drum. hand drum. djembe. "
    "tambourine. cymbal. piano. keyboard. microphone. cushion. throw pillow. "
    "sofa. couch. armchair. chair. ottoman. picture frame. painting. map. "
    "poster. mirror. lamp. plant. book. mug. cup. bottle. can. plate. bowl. "
    "remote. phone. laptop. notebook. hand. arm. person. face. box. bag. hat. "
    "ball. toy. blanket. towel. shoe. object."
)

VALID_DETECTORS = ("dino", "yolo", "mediapipe")
DEFAULT_DETECTOR = "dino"


def _runtime_dir() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))),
        "runtime",
    )


def _active_detector_path() -> str:
    return os.path.join(_runtime_dir(), "active_detector.json")


def read_active_detector() -> str:
    """Read the currently-selected detector name from disk.

    Returns DEFAULT_DETECTOR if the file is missing, malformed, or names
    an unknown backend.
    """
    path = _active_detector_path()
    try:
        with open(path) as f:
            data = json.load(f)
        name = str(data.get("detector", DEFAULT_DETECTOR)).lower()
        if name in VALID_DETECTORS:
            return name
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return DEFAULT_DETECTOR


def write_active_detector(name: str) -> None:
    name = name.lower()
    if name not in VALID_DETECTORS:
        raise ValueError(f"unknown detector {name!r}; choices: {VALID_DETECTORS}")
    path = _active_detector_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"detector": name}, f, indent=2)


@dataclass
class RecognitionResult:
    sam_mask_cam: np.ndarray          # uint8 {0,255}, camera-space, refined mask
    label: str                        # best detector label
    label_score: float                # 0..1
    sam_score: float                  # SAM's own quality score
    source_blob_id: int


# =========================================================================
# Base: SAM2 segmentation (shared by all three detector backends).
# =========================================================================
class _BaseRecognizer:
    """Loads SAM2 once; subclasses add their own detector for `label_image`."""

    def __init__(
        self,
        sam_checkpoint: str = "facebook/sam2.1-hiera-large",
        device: Optional[str] = None,
        dtype: str = "float16",
    ):
        import torch                                    # heavy import deferred
        from transformers import AutoProcessor, Sam2Model

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, dtype)

        print(f"[recognize] loading SAM2 ({sam_checkpoint}) on {self.device}...")
        t0 = time.perf_counter()
        self.sam_processor = AutoProcessor.from_pretrained(sam_checkpoint)
        self.sam_model = Sam2Model.from_pretrained(sam_checkpoint).to(self.device)
        self.sam_model.eval()
        print(f"[recognize] SAM2 loaded in {time.perf_counter()-t0:.1f}s")

    @property
    def torch(self):
        return self._torch

    # ----- SAM 2 ---------------------------------------------------------------
    def segment_with_points(
        self, color_bgr: np.ndarray, points_cam: List[Tuple[float, float]]
    ) -> List[Tuple[np.ndarray, float]]:
        """For each (x, y) point in camera pixels, return (mask uint8, score)."""
        torch = self._torch
        if not points_cam:
            return []
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        input_points = [[[[float(px), float(py)]] for (px, py) in points_cam]]
        input_labels = [[[1] for _ in points_cam]]
        inputs = self.sam_processor(
            images=rgb, input_points=input_points, input_labels=input_labels,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.sam_model(**inputs, multimask_output=True)
        masks = self.sam_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
        )[0]
        scores_all = outputs.iou_scores[0].cpu().numpy()
        results: List[Tuple[np.ndarray, float]] = []
        for i in range(len(points_cam)):
            scores = scores_all[i]
            best = int(np.argmax(scores))
            m = masks[i, best].numpy().astype(np.uint8) * 255
            results.append((m, float(scores[best])))
        return results

    # ----- SAM 2 automatic mask generator (for splitting mega-blobs) ----------
    def auto_segment_in_bbox(
        self,
        color_bgr: np.ndarray,
        bbox_xywh: Tuple[int, int, int, int],
        parent_mask: Optional[np.ndarray] = None,
        grid_n: int = 4,
        score_thresh: float = 0.55,
        dedupe_iou: float = 0.55,
        max_area_frac: float = 0.85,
        min_area_px: int = 800,
    ) -> List[Tuple[np.ndarray, float]]:
        x, y, w, h = bbox_xywh
        if w <= 0 or h <= 0:
            return []
        H_full, W_full = color_bgr.shape[:2]
        parent_area = int((parent_mask > 0).sum()) if parent_mask is not None else (w * h)
        if parent_area <= 0:
            return []
        inset_x = w / (grid_n + 1)
        inset_y = h / (grid_n + 1)
        points: List[Tuple[float, float]] = []
        for gy in range(grid_n):
            for gx in range(grid_n):
                px = x + inset_x * (gx + 1)
                py = y + inset_y * (gy + 1)
                if not (0 <= px < W_full and 0 <= py < H_full):
                    continue
                if parent_mask is not None and parent_mask[int(py), int(px)] == 0:
                    continue
                points.append((px, py))
        if not points:
            return []
        raw = self.segment_with_points(color_bgr, points)
        candidates: List[Tuple[np.ndarray, float]] = []
        for mask, score in raw:
            if score < score_thresh:
                continue
            area = int((mask > 0).sum())
            if area < min_area_px:
                continue
            if area > parent_area * max_area_frac:
                continue
            if parent_mask is not None:
                inside = int(np.logical_and(mask > 0, parent_mask > 0).sum())
                if inside / max(1, area) < 0.5:
                    continue
            candidates.append((mask, score))
        candidates.sort(key=lambda t: -t[1])
        kept: List[Tuple[np.ndarray, float]] = []
        for mask, score in candidates:
            ok = True
            for kmask, _ in kept:
                inter = int(np.logical_and(mask > 0, kmask > 0).sum())
                union = int(np.logical_or(mask > 0, kmask > 0).sum())
                if union > 0 and inter / union > dedupe_iou:
                    ok = False
                    break
            if ok:
                kept.append((mask, score))
        return kept

    # ----- detector interface — each subclass overrides label_image ----------
    def label_image(self, color_bgr: np.ndarray, **kwargs) -> List[dict]:
        raise NotImplementedError("subclass must implement label_image()")

    # ----- combined (kept for any callers; not used by the live pipeline) ----
    def recognize(
        self,
        color_bgr: np.ndarray,
        blobs: List[Blob],
        prompt: str = DEFAULT_DINO_PROMPT,
    ) -> List[RecognitionResult]:
        results: List[RecognitionResult] = []
        if not blobs:
            return results
        sam_out = self.segment_with_points(
            color_bgr, [b.centroid_cam for b in blobs]
        )
        dets = self.label_image(color_bgr, prompt=prompt)
        for blob, (sam_mask, sam_score) in zip(blobs, sam_out):
            bx, by, bw, bh = _mask_bbox(sam_mask)
            best_label = "object"
            best_score = 0.0
            for det in dets:
                iou = _bbox_iou((bx, by, bw, bh), det["bbox"])
                weighted = iou * det["score"]
                if weighted > best_score:
                    best_score = weighted
                    best_label = det["label"] or "object"
            results.append(RecognitionResult(
                sam_mask_cam=sam_mask,
                label=best_label,
                label_score=best_score,
                sam_score=sam_score,
                source_blob_id=blob.blob_id,
            ))
        return results


# =========================================================================
# Backend 1: Grounding DINO Tiny (open-vocab via text prompt).
# =========================================================================
class DinoRecognizer(_BaseRecognizer):
    name = "dino"

    def __init__(
        self,
        sam_checkpoint: str = "facebook/sam2.1-hiera-large",
        dino_checkpoint: str = "IDEA-Research/grounding-dino-tiny",
        device: Optional[str] = None,
        dtype: str = "float16",
    ):
        super().__init__(sam_checkpoint=sam_checkpoint, device=device, dtype=dtype)
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        print(f"[recognize] loading Grounding DINO ({dino_checkpoint})...")
        t0 = time.perf_counter()
        self.dino_processor = AutoProcessor.from_pretrained(dino_checkpoint)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            dino_checkpoint
        ).to(self.device)
        self.dino_model.eval()
        print(f"[recognize] DINO loaded in {time.perf_counter()-t0:.1f}s")

    def label_image(
        self,
        color_bgr: np.ndarray,
        prompt: str = DEFAULT_DINO_PROMPT,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        **_,
    ) -> List[dict]:
        torch = self._torch
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.dino_processor(
            images=rgb, text=prompt, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.dino_model(**inputs)
        H, W = rgb.shape[:2]
        target_sizes = torch.tensor([[H, W]], device=self.device)
        post = self.dino_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=target_sizes,
        )[0]
        out = []
        label_list = post.get("text_labels", post.get("labels", []))
        for score, label, box in zip(
            post["scores"].cpu().numpy(),
            label_list,
            post["boxes"].cpu().numpy(),
        ):
            x0, y0, x1, y1 = box
            out.append({
                "label": str(label),
                "score": float(score),
                "bbox": (int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
            })
        return out


# =========================================================================
# Backend 2: Ultralytics YOLO11n (closed-vocab COCO).
# =========================================================================
class YoloRecognizer(_BaseRecognizer):
    name = "yolo"

    def __init__(
        self,
        sam_checkpoint: str = "facebook/sam2.1-hiera-large",
        yolo_weights: str = "yolo11n.pt",
        device: Optional[str] = None,
        dtype: str = "float16",
    ):
        super().__init__(sam_checkpoint=sam_checkpoint, device=device, dtype=dtype)
        from ultralytics import YOLO
        print(f"[recognize] loading YOLO ({yolo_weights})...")
        t0 = time.perf_counter()
        # ultralytics auto-downloads the weight on first use. To keep the
        # cache scoped to this project, switch CWD into runtime/models for
        # the load; the file lands there and subsequent loads find it.
        models_dir = os.path.join(_runtime_dir(), "models")
        os.makedirs(models_dir, exist_ok=True)
        target = os.path.join(models_dir, yolo_weights)
        # If the file already lives in models_dir, load by absolute path.
        # Otherwise let ultralytics download into models_dir.
        if os.path.exists(target):
            self.yolo_model = YOLO(target)
        else:
            cwd = os.getcwd()
            try:
                os.chdir(models_dir)
                self.yolo_model = YOLO(yolo_weights)
            finally:
                os.chdir(cwd)
        # Warm the model on a tiny image so first real frame isn't slow.
        try:
            self.yolo_model.to(self.device)
        except Exception:
            pass
        print(f"[recognize] YOLO loaded in {time.perf_counter()-t0:.1f}s "
              f"(device={self.device}, classes={len(self.yolo_model.names)})")

    def label_image(
        self,
        color_bgr: np.ndarray,
        prompt: str = "",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        **_,
    ) -> List[dict]:
        # YOLO ignores the text prompt — vocabulary is fixed (COCO 80).
        # We map box_threshold -> conf so the existing pipeline knob still
        # has an effect.
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        results = self.yolo_model.predict(
            rgb, conf=float(box_threshold), device=self.device,
            verbose=False,
        )
        out: List[dict] = []
        if not results:
            return out
        r = results[0]
        names = r.names  # dict[int -> str]
        if r.boxes is None or r.boxes.xyxy is None:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        for (x0, y0, x1, y1), conf, c in zip(xyxy, confs, clss):
            out.append({
                "label": str(names.get(int(c), f"class_{int(c)}")),
                "score": float(conf),
                "bbox": (int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
            })
        return out


# =========================================================================
# Backend 3: MediaPipe Object Detector EfficientDet-Lite0 (closed-vocab COCO).
# =========================================================================
_MP_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/object_detector/"
    "efficientdet_lite0/float32/1/efficientdet_lite0.tflite"
)


class MediapipeRecognizer(_BaseRecognizer):
    name = "mediapipe"

    def __init__(
        self,
        sam_checkpoint: str = "facebook/sam2.1-hiera-large",
        mp_model_url: str = _MP_MODEL_URL,
        device: Optional[str] = None,
        dtype: str = "float16",
    ):
        super().__init__(sam_checkpoint=sam_checkpoint, device=device, dtype=dtype)
        import mediapipe as mp
        from mediapipe.tasks import python as mp_py
        from mediapipe.tasks.python import vision as mp_vision

        models_dir = os.path.join(_runtime_dir(), "models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, "efficientdet_lite0.tflite")
        if not os.path.exists(model_path):
            print(f"[recognize] downloading MediaPipe model -> {model_path}")
            urllib.request.urlretrieve(mp_model_url, model_path)

        print(f"[recognize] loading MediaPipe ObjectDetector ({model_path})...")
        t0 = time.perf_counter()
        base_opts = mp_py.BaseOptions(model_asset_path=model_path)
        # MediaPipe's TFLite delegate is CPU by default; GPU delegate
        # requires extra build flags on Windows, so leave it on CPU.
        opts = mp_vision.ObjectDetectorOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,
            score_threshold=0.25,
            max_results=50,
        )
        self.mp_module = mp
        self.mp_detector = mp_vision.ObjectDetector.create_from_options(opts)
        print(f"[recognize] MediaPipe loaded in {time.perf_counter()-t0:.1f}s")

    def label_image(
        self,
        color_bgr: np.ndarray,
        prompt: str = "",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        **_,
    ) -> List[dict]:
        mp = self.mp_module
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.mp_detector.detect(mp_image)
        out: List[dict] = []
        if not result or not result.detections:
            return out
        for det in result.detections:
            if not det.categories:
                continue
            cat = det.categories[0]
            if cat.score < float(box_threshold):
                continue
            bb = det.bounding_box  # origin_x, origin_y, width, height
            out.append({
                "label": str(cat.category_name or f"class_{cat.index}"),
                "score": float(cat.score),
                "bbox": (int(bb.origin_x), int(bb.origin_y),
                         int(bb.width), int(bb.height)),
            })
        return out


# =========================================================================
# Factory.
# =========================================================================
def create_recognizer(name: Optional[str] = None, **kwargs) -> _BaseRecognizer:
    """Build the recognizer named by `name` (or by runtime/active_detector.json)."""
    if name is None:
        name = read_active_detector()
    name = name.lower()
    if name == "dino":
        return DinoRecognizer(**kwargs)
    if name == "yolo":
        return YoloRecognizer(**kwargs)
    if name == "mediapipe":
        return MediapipeRecognizer(**kwargs)
    raise ValueError(f"unknown detector {name!r}; choices: {VALID_DETECTORS}")


# Back-compat alias: any existing code that says `Recognizer()` still gets
# DINO (which was the only option before).
Recognizer = DinoRecognizer


# ---- helpers --------------------------------------------------------------
def _mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def _bbox_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ix0 = max(ax, bx); iy0 = max(ay, by)
    ix1 = min(ax + aw, bx + bw); iy1 = min(ay + ah, by + bh)
    iw = max(0, ix1 - ix0); ih = max(0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0
