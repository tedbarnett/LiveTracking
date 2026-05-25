"""Cobblestone pattern v4 + v5 - matches deep-joint running-bond reference.

v4: realistic look, neutral grey palette, for production idle.
v5: same look but each stone has a subtle calibration-color tint, so the
    decoder can still identify it. Audience sees a cobblestone street;
    camera sees a calibration target.
"""
import os
import sys
import random
import numpy as np
import cv2


W, H = 1920, 1080
MORTAR_BGR = (38, 38, 42)            # dark joint color
BG_PADDING = (28, 26, 28)            # canvas behind everything

# Grey palette anchored on reference: RGB(135,130,128), highlights +30, shadow -40
GREY_STONE_BGRS = [
    (115, 115, 120), (125, 125, 130), (135, 135, 140), (130, 132, 138),
    (140, 142, 148), (145, 145, 150), (118, 122, 130), (128, 132, 138),
    (138, 140, 146), (148, 150, 155), (122, 126, 134), (132, 136, 142),
    (110, 112, 118), (138, 138, 142), (125, 130, 138), (142, 144, 148),
]

# Vivid tints used in v5 for calibration encoding (kept subtle - mixed 20%
# with grey so it reads cobblestone first, calibration second)
CALIBRATION_TINTS = [
    (255,  60,  60), ( 60, 255,  60), ( 60,  60, 255), (255, 255,  60),
    (255,  60, 255), ( 60, 255, 255), (255, 150,  60), (150,  60, 255),
    ( 60, 255, 150), (255,  60, 150), ( 60, 150, 255), (150, 255,  60),
    (200, 100,  50), ( 50, 200, 100), (100,  50, 200), (200,  50, 100),
    ( 50, 100, 200), (100, 200,  50), (220, 220,  80), (220,  80, 220),
    ( 80, 220, 220), (180, 120,  60), (120,  60, 180), ( 60, 180, 120),
    (240, 160,  40), (160,  40, 240), ( 40, 240, 160), (240,  40, 160),
    ( 40, 160, 240), (160, 240,  40), (200, 180, 100), (180, 100, 200),
    (100, 200, 180), (220, 140,  70), (140,  70, 220), ( 70, 220, 140),
    (180,  80, 100), ( 80, 100, 180), (100, 180,  80), (200, 120, 140),
    (120, 140, 200), (140, 200, 120), (160, 100, 220), (100, 220, 160),
    (220, 160, 100), (140, 180,  60), (180,  60, 140), ( 60, 140, 180),
]


def make_rounded_rect_polygon(cx, cy, w, h, corner_radius, jitter=0.06,
                               n_per_side=6, rng=None):
    """Build a polygon approximating a tumbled/rounded square stone.

    Walks the perimeter sampling points along each side and rounding corners.
    Adds small random jitter for the chipped/weathered look.
    """
    if rng is None:
        rng = random.Random()
    w2 = w / 2.0
    h2 = h / 2.0
    cr = corner_radius

    pts = []

    def add_corner(corner_cx, corner_cy, ang_start, ang_end):
        for i in range(n_per_side):
            t = i / (n_per_side - 1)
            ang = ang_start + (ang_end - ang_start) * t
            # jitter radius per point so the corner isn't perfectly smooth
            r = cr * (1.0 + rng.uniform(-jitter, jitter))
            x = corner_cx + r * np.cos(ang)
            y = corner_cy + r * np.sin(ang)
            pts.append([x, y])

    # corners: top-right, bottom-right, bottom-left, top-left
    add_corner(cx + w2 - cr, cy - h2 + cr, -np.pi / 2, 0)
    add_corner(cx + w2 - cr, cy + h2 - cr, 0, np.pi / 2)
    add_corner(cx - w2 + cr, cy + h2 - cr, np.pi / 2, np.pi)
    add_corner(cx - w2 + cr, cy - h2 + cr, np.pi, 3 * np.pi / 2)

    # global perimeter jitter for chipped edges
    pts = np.array(pts, dtype=np.float32)
    for i in range(len(pts)):
        pts[i, 0] += rng.uniform(-w * 0.012, w * 0.012)
        pts[i, 1] += rng.uniform(-h * 0.012, h * 0.012)

    return np.round(pts).astype(np.int32)


def add_stone_texture(canvas, polygon, base_color, rng):
    """Add weathered surface texture: subtle highlight, shadow, and speckle."""
    # bounding region
    x, y, w, h = cv2.boundingRect(polygon)
    if w < 4 or h < 4:
        return

    # gradient highlight from top-left
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon - np.array([[x, y]])], 255)

    # local subcanvas
    sub = canvas[y:y + h, x:x + w].astype(np.float32)

    # vertical light gradient: brighter top, darker bottom
    grad = np.linspace(1.18, 0.82, h, dtype=np.float32)[:, None]
    grad = np.repeat(grad, w, axis=1)

    # slight horizontal cross-light
    grad *= np.linspace(1.06, 0.94, w, dtype=np.float32)[None, :]

    # apply only inside mask
    m = (mask > 0)
    sub_lit = sub.copy()
    sub_lit[m] = np.clip(sub[m] * grad[m, None], 0, 255)

    canvas[y:y + h, x:x + w] = sub_lit.astype(np.uint8)

    # speckle/pitting - dark micro-spots for weathered surface
    n_specks = rng.randint(8, 20)
    cx_local = w // 2
    cy_local = h // 2
    for _ in range(n_specks):
        sx = rng.randint(0, w - 1)
        sy = rng.randint(0, h - 1)
        if mask[sy, sx] == 0:
            continue
        r = rng.randint(1, 3)
        darken = rng.uniform(0.55, 0.85)
        cv2.circle(canvas, (x + sx, y + sy), r,
                   tuple(int(c * darken) for c in base_color), -1, cv2.LINE_AA)

    # occasional tiny highlight specks (mineral fleck)
    n_highs = rng.randint(2, 6)
    for _ in range(n_highs):
        sx = rng.randint(0, w - 1)
        sy = rng.randint(0, h - 1)
        if mask[sy, sx] == 0:
            continue
        cv2.circle(canvas, (x + sx, y + sy), 1, (200, 200, 205), -1, cv2.LINE_AA)


def render_cobblestone_realistic(out_path, calibration_mode=False,
                                  tint_strength=0.0, label=""):
    """Render running-bond cobblestone pattern with deep mortar joints.

    calibration_mode=True: each stone gets a unique calibration tint (tint_strength controls how visible).
    """
    rng = random.Random(42)

    canvas = np.full((H, W, 3), BG_PADDING, dtype=np.uint8)

    # Running bond layout: rows of ~120-150 px stones, offset every other row
    target_stone_h = 110
    target_stone_w = 140
    mortar_gap = 12

    margin = 30
    usable_h = H - 2 * margin
    n_rows = int(round(usable_h / (target_stone_h + mortar_gap)))
    actual_row_h = usable_h / n_rows

    stone_idx = 0  # for calibration encoding

    for row in range(n_rows):
        y_center = int(margin + actual_row_h * (row + 0.5))
        # offset every other row by half a stone width
        row_offset = (target_stone_w + mortar_gap) / 2 if row % 2 == 1 else 0
        # how many stones fit in this row?
        usable_w = W - 2 * margin
        n_stones = int(round(usable_w / (target_stone_w + mortar_gap)))
        # slight per-row size variation so it doesn't look CAD-perfect
        row_size_factor = 1.0 + rng.uniform(-0.05, 0.05)
        stone_w_row = target_stone_w * row_size_factor
        stone_h_row = (actual_row_h - mortar_gap) * 0.95

        x_start = margin + row_offset
        # if offset pushes us off the right edge, drop one stone
        if x_start + n_stones * (stone_w_row + mortar_gap) > W - margin:
            n_stones -= 1

        for col in range(n_stones):
            x_center = int(x_start + (stone_w_row + mortar_gap) * (col + 0.5))

            # per-stone tiny size variation
            sw = stone_w_row * (1.0 + rng.uniform(-0.05, 0.05))
            sh = stone_h_row * (1.0 + rng.uniform(-0.04, 0.04))

            corner_r = min(sw, sh) * rng.uniform(0.18, 0.30)
            poly = make_rounded_rect_polygon(
                x_center, y_center, sw, sh, corner_r,
                jitter=rng.uniform(0.08, 0.18), rng=rng
            )

            # base color
            grey_color = GREY_STONE_BGRS[rng.randint(0, len(GREY_STONE_BGRS) - 1)]

            if calibration_mode and tint_strength > 0:
                tint = CALIBRATION_TINTS[stone_idx % len(CALIBRATION_TINTS)]
                base = tuple(
                    int(grey_color[i] * (1 - tint_strength) + tint[i] * tint_strength)
                    for i in range(3)
                )
            else:
                base = grey_color

            # subtle shadow into the joint
            shadow_poly = poly + np.array([[3, 4]], dtype=np.int32)
            cv2.fillPoly(canvas, [shadow_poly], (15, 15, 17))

            # fill the stone
            cv2.fillPoly(canvas, [poly], base)

            # rim darken (deep joint shadow)
            rim_color = tuple(int(c * 0.55) for c in base)
            cv2.polylines(canvas, [poly], isClosed=True, color=rim_color,
                          thickness=2, lineType=cv2.LINE_AA)

            # add weathered texture
            add_stone_texture(canvas, poly, base, rng)

            stone_idx += 1

    # Label
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, H - 70), (W, H), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, dst=canvas)
    cv2.putText(canvas, label, (40, H - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(canvas, "running-bond, deep-joint, weathered",
                (W - 600, H - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas)
    print(f"wrote {out_path}")


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # v4: pure realistic, no calibration tint, production idle look
    render_cobblestone_realistic(
        os.path.join(out_dir, "cobble_v4_realistic.png"),
        calibration_mode=False,
        tint_strength=0.0,
        label="v4_REALISTIC - production idle, matches reference",
    )

    # v5: same look, subtle calibration tint visible to camera but not audience
    render_cobblestone_realistic(
        os.path.join(out_dir, "cobble_v5_realistic_calibration_subtle.png"),
        calibration_mode=True,
        tint_strength=0.18,  # 18% tint - audience reads it as cobblestone, camera sees colors
        label="v5_CALIBRATION_SUBTLE - 18% calibration tint mixed into stones",
    )

    # v6: same look but stronger tint for tougher decode conditions
    render_cobblestone_realistic(
        os.path.join(out_dir, "cobble_v6_realistic_calibration_strong.png"),
        calibration_mode=True,
        tint_strength=0.50,  # 50% tint - clearly colorful, still cobblestone-shaped
        label="v6_CALIBRATION_STRONG - 50% tint, calibration-mode flash",
    )

    print("done")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
