"""Edge-based detection.

Two stages:
1. Find the projection rectangle on the wall using Canny edges + contour fitting.
2. Within that rectangle, find the post-it rectangles using Canny edges +
   contour fitting + size filter (76mm post-its at ~1.5m camera distance
   should be ~60-100 px square).

No brightness percentile games. No depth tricks. Just geometry.

Output: annotated image showing detected rectangles, plus calibration data.
"""
import os
import sys
import time
import json
import math
import numpy as np
import cv2
import pygame
import pyrealsense2 as rs


DISPLAY_X = 5120
DISPLAY_Y = 0
DISPLAY_W = 1280
DISPLAY_H = 720


def capture_under_white_flood():
    """Project white, capture aligned RGB frame from D455."""
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{DISPLAY_X},{DISPLAY_Y}"
    pygame.init()
    screen = pygame.display.set_mode((DISPLAY_W, DISPLAY_H), pygame.NOFRAME)
    screen.fill((255, 255, 255))
    pygame.display.flip()
    # pump
    for _ in range(15):
        for _ in pygame.event.get():
            pass
        time.sleep(0.03)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipe.start(cfg)
    try:
        for _ in range(25):
            pipe.wait_for_frames()
        frames = pipe.wait_for_frames()
        cam_img = np.asanyarray(frames.get_color_frame().get_data())
    finally:
        pipe.stop()
    # Don't quit pygame so the white projection stays on
    return cam_img, screen


def find_projection_quad(cam_img):
    """Find the projection rectangle on the wall. Returns the 4 corners
    as a (4, 2) array of (x, y), or None if not found."""
    gray = cv2.cvtColor(cam_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # The projection is a BRIGHT region. We want its outer edge.
    # Use Otsu threshold to separate bright from dark.
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Clean up small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=2)

    # Largest connected component = the projection
    num, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    if num <= 1:
        return None, bw
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = ((lab == biggest).astype(np.uint8)) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    cnt = max(contours, key=cv2.contourArea)

    # Approximate to a quadrilateral
    peri = cv2.arcLength(cnt, True)
    for eps_factor in [0.02, 0.03, 0.04, 0.05, 0.06]:
        approx = cv2.approxPolyDP(cnt, eps_factor * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2), mask
    # If we can't find a 4-vertex approx, use the bounding box
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.int32)
    return box, mask


def find_postits_inside(cam_img, projection_mask):
    """Find post-it-sized roughly-square contours inside the projection.

    Approach: within the projection mask, Canny edges to outline the post-its,
    then findContours, then filter by:
        - area in 1500..15000 px (post-its at ~1.5m are ~60-100 px square)
        - aspect ratio < 1.6
        - polygon approximation has 4 vertices (rectangular)
        - centroid inside the projection mask
    """
    # Restrict to projection ROI
    roi_img = cv2.bitwise_and(cam_img, cam_img, mask=projection_mask)
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold is robust to projection lighting unevenness
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 35, 110)
    # Connect edge gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    postits = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < 800 or a > 30000:
            continue
        peri = cv2.arcLength(c, True)
        if peri < 50:
            continue
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) < 4 or len(approx) > 6:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if min(w, h) < 15:
            continue
        aspect = max(w, h) / min(w, h)
        if aspect > 1.8:
            continue
        # Centroid must be inside the projection mask
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        if projection_mask[cy, cx] == 0:
            continue
        # Solidity check: filled area / hull area > 0.8 for clean rectangles
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull) or 1
        solidity = a / hull_area
        if solidity < 0.8:
            continue
        postits.append({
            "centroid": (cx, cy),
            "bbox": (int(x), int(y), int(w), int(h)),
            "area": float(a),
            "size_px": max(int(w), int(h)),
            "n_vertices": int(len(approx)),
            "aspect": float(aspect),
            "solidity": float(solidity),
        })

    # Dedup overlapping contours (Canny often gives inner+outer edge of the
    # same post-it). Group by centroid proximity, keep largest of each group.
    deduped = []
    used = set()
    for i, p in enumerate(postits):
        if i in used:
            continue
        cluster = [p]
        for j in range(i + 1, len(postits)):
            if j in used:
                continue
            q = postits[j]
            dx = p["centroid"][0] - q["centroid"][0]
            dy = p["centroid"][1] - q["centroid"][1]
            if dx * dx + dy * dy < 35 * 35:
                cluster.append(q)
                used.add(j)
        used.add(i)
        # Keep biggest of the cluster
        cluster.sort(key=lambda x: -x["area"])
        deduped.append(cluster[0])

    return deduped


def main():
    print("capturing under white flood...", flush=True)
    cam_img, screen = capture_under_white_flood()
    cv2.imwrite(r"D:\Github-D\LiveTracking\tmp\edge_capture.png", cam_img)

    print("finding projection quadrilateral...", flush=True)
    quad, proj_mask = find_projection_quad(cam_img)
    if quad is None:
        print("NO PROJECTION QUAD FOUND", flush=True)
    else:
        print(f"projection quad corners: {quad.tolist()}", flush=True)

    cv2.imwrite(r"D:\Github-D\LiveTracking\tmp\edge_proj_mask.png", proj_mask)

    if proj_mask is None or quad is None:
        pygame.quit()
        return 1

    print("finding post-its inside projection...", flush=True)
    postits = find_postits_inside(cam_img, proj_mask)
    print(f"found {len(postits)} post-it candidate(s)", flush=True)
    for i, p in enumerate(postits):
        print(f"  #{i+1}: centroid={p['centroid']} bbox={p['bbox']} "
              f"size={p['size_px']}px solidity={p['solidity']:.2f}", flush=True)

    # Estimate depth using known post-it size (76mm) and approximate focal length
    FX = 642.0  # D455 1280x720 approx focal length in pixels
    POSTIT_MM = 76.0
    for p in postits:
        depth_mm = FX * POSTIT_MM / p["size_px"]
        p["estimated_depth_mm"] = float(depth_mm)
        print(f"    estimated depth: {depth_mm:.0f}mm", flush=True)

    # Annotate
    out = cam_img.copy()
    cv2.polylines(out, [quad.reshape(-1, 1, 2)], True, (0, 255, 255), 3)
    cv2.putText(out, "projection rect", (quad[0][0] + 8, quad[0][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    for i, p in enumerate(postits):
        x, y, w, h = p["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(out, f"#{i+1} {p['size_px']}px {p.get('estimated_depth_mm', 0):.0f}mm",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imwrite(r"D:\Github-D\LiveTracking\tmp\edge_annotated.png", out)
    print("annotated image saved to tmp/edge_annotated.png", flush=True)

    # Dump JSON
    result = {
        "projection_quad": quad.tolist(),
        "postits": [
            {k: v for k, v in p.items()}
            for p in postits
        ],
        "fx_px": FX,
        "postit_size_mm": POSTIT_MM,
    }
    with open(r"D:\Github-D\LiveTracking\tmp\edge_result.json", "w") as f:
        json.dump(result, f, indent=2)

    # Hold the white flood so it stays consistent
    print("press Q/Esc to quit (or close this script)", flush=True)
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key in (pygame.K_q, pygame.K_ESCAPE):
                running = False
        time.sleep(0.03)
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
