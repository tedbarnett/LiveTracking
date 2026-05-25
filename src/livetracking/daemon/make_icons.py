"""Generate favicon + home-screen icons from cobblestone-source.jpg.

Run once whenever the source art changes.
"""
import os
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
SRC = os.path.join(STATIC, "cobblestone-source.jpg")


def center_square(img):
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def main():
    img = Image.open(SRC).convert("RGB")
    sq = center_square(img)
    # Build the multi-resolution .ico (16, 32, 48, 64, 128, 256)
    ico_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    ico_path = os.path.join(STATIC, "favicon.ico")
    sq.save(ico_path, format="ICO", sizes=ico_sizes)
    print(f"wrote {ico_path}")
    # Home-screen icons - PNG, square
    for size in [180, 192, 256, 384, 512]:
        out = os.path.join(STATIC, f"icon-{size}.png")
        sq.resize((size, size), Image.LANCZOS).save(out, optimize=True)
        print(f"wrote {out}")
    # Also a single 32x32 PNG as a fallback some browsers prefer
    out = os.path.join(STATIC, "favicon-32.png")
    sq.resize((32, 32), Image.LANCZOS).save(out, optimize=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
