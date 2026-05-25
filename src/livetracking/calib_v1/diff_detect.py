"""Ambient-independent post-it detection via projector differencing.

Why this exists
---------------
The original edge_detect.py + closed_loop_search.py worked great at night
(2026-05-24 ~11:35 PM ET) with room lights only and a fixed RealSense
exposure of 150. The next morning (~8:30 AM ET), with the same physical
scene but daylight added, Otsu thresholding on the white-flood camera frame
broke: it could no longer separate the projection rectangle from the
sunlit wall. Detection found 0 post-its.

The fix here works regardless of ambient brightness because it doesn't
depend on raw pixel values. It uses the principle your eye uses:
"the projection is what changed when the projector turned on."

Algorithm
---------
1. Project pure black -> capture cam_black.
2. Project pure white -> capture cam_white.
3. diff = clip(cam_white_gray - cam_black_gray, 0, 255). Threshold above
   3x camera noise floor (typical: thresh=30). That mask IS the
   projection rectangle, no matter what the ambient looks like.
4. Inside the projection mask, find the brightest ~8% of pixels.
   Post-its reflect more projector light than the surrounding wall, so
   they sit in the top quantile. Use the 92nd percentile as the
   threshold (adaptive, robust to projector brightness).
5. Connected components within that top-quantile mask, filtered by area
   and aspect ratio, are the post-its.
6. Per-blob minAreaRect captures size + rotation for the renderer.

Lesson
------
Don't rely on absolute pixel thresholds for detection - they're brittle
to ambient. Use what the projector adds, not what the camera sees.

Original v8 detector (Otsu+morphology+Canny) is preserved in edge_detect.py
for the dark-room case where it works fine. This is the morning-and-bright
fallback.
"""
import cv2
import numpy as np


def find_postits_diff(cam_black, cam_white,
                       proj_thresh=30,
                       top_quantile=0.92,
                       min_area=80,
                       max_area=8000,
                       max_aspect=3.0,
                       min_proj_pixels=5000):
    """Detect post-its from black-vs-white projector capture pair.

    Parameters
    ----------
    cam_black, cam_white : np.ndarray
        BGR camera frames captured while projector showed pure black /
        pure white respectively.
    proj_thresh : int
        Diff threshold for "projector is here" mask. Default 30 = ~3x
        typical RealSense noise floor at exposure=150.
    top_quantile : float
        Brightness percentile inside the projection mask above which
        pixels are treated as post-it candidates. 0.92 = top 8%.
    min_area, max_area : int
        Connected-component area filter (camera pixels).
    max_aspect : float
        Max width/height ratio. 3.0 allows for moderately elongated
        post-its (rotated rectangles get long bounding boxes).
    min_proj_pixels : int
        If the diff mask has fewer pixels than this, the projector
        probably isn't being seen by the camera. Returns None.

    Returns
    -------
    list of dict or None
        Each dict: {"centroid_cam": (cx, cy), "rot_size": [w, h],
        "rot_angle_deg": deg, "area": area}. None if fewer than 3
        candidates pass filtering.
    """
    bg = cv2.cvtColor(cam_black, cv2.COLOR_BGR2GRAY).astype(np.int16)
    wg = cv2.cvtColor(cam_white, cv2.COLOR_BGR2GRAY).astype(np.int16)
    diff = np.clip(wg - bg, 0, 255).astype(np.uint8)

    _, proj_mask = cv2.threshold(diff, proj_thresh, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    proj_mask = cv2.morphologyEx(proj_mask, cv2.MORPH_CLOSE, k, iterations=2)
    proj_mask = cv2.morphologyEx(proj_mask, cv2.MORPH_OPEN, k, iterations=1)
    if proj_mask.sum() < min_proj_pixels:
        return None

    # Keep only the biggest connected component of the projection mask
    # (suppresses speckle outside the main rectangle).
    num, lab, stats, _ = cv2.connectedComponentsWithStats(proj_mask, 8)
    if num <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    proj_mask = ((lab == biggest).astype(np.uint8)) * 255

    inside = cv2.cvtColor(cam_white, cv2.COLOR_BGR2GRAY)
    pixels_in_proj = inside[proj_mask > 0]
    if len(pixels_in_proj) < 1000:
        return None

    thresh_val = float(np.percentile(pixels_in_proj, top_quantile * 100))
    bright = ((inside >= thresh_val) & (proj_mask > 0)).astype(np.uint8) * 255
    bright = cv2.morphologyEx(
        bright, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1)

    num2, lab2, stats2, cents2 = cv2.connectedComponentsWithStats(bright, 8)
    cands = []
    for i in range(1, num2):
        a = int(stats2[i, cv2.CC_STAT_AREA])
        if a < min_area or a > max_area:
            continue
        bw_ = stats2[i, cv2.CC_STAT_WIDTH]
        bh_ = stats2[i, cv2.CC_STAT_HEIGHT]
        if max(bw_, bh_) / max(1, min(bw_, bh_)) > max_aspect:
            continue
        cx, cy = float(cents2[i][0]), float(cents2[i][1])
        ys, xs = np.where(lab2 == i)
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        (rcx, rcy), (rw, rh), rangle = cv2.minAreaRect(pts)
        cands.append({
            "centroid_cam": (cx, cy),
            "rot_size": [float(rw), float(rh)],
            "rot_angle_deg": float(rangle),
            "area": float(a),
        })

    # Cluster close duplicates (a single post-it occasionally fragments).
    deduped = []
    used = set()
    for i, p in enumerate(cands):
        if i in used:
            continue
        cluster = [p]
        for j in range(i + 1, len(cands)):
            if j in used:
                continue
            q = cands[j]
            dx = p["centroid_cam"][0] - q["centroid_cam"][0]
            dy = p["centroid_cam"][1] - q["centroid_cam"][1]
            if dx * dx + dy * dy < 30 * 30:
                cluster.append(q)
                used.add(j)
        used.add(i)
        cluster.sort(key=lambda x: -x["area"])
        deduped.append(cluster[0])

    deduped.sort(key=lambda x: -x["area"])
    deduped = deduped[:3]
    deduped.sort(key=lambda x: x["centroid_cam"][0])
    return deduped if len(deduped) >= 3 else None
