"""Manual square-crop tool for the Citrus Fusion AI dataset.

Run at the repository root:
    py scripts/manual_square_crop.py

Controls:
    Drag with left mouse button : move square (or reposition it by clicking outside)
    Mouse wheel / + / -        : enlarge / shrink square
    Enter                      : crop, resize, and save as RGB PNG; next image
    N                          : skip image
    R                          : reset square
    Q or Esc                   : quit safely

Output: 224x224 RGB PNG (lossless) in dataset/processed, preserving folder structure.
Original images in dataset/raw are never changed.
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "dataset" / "raw"
PROCESSED_DIR = BASE_DIR / "dataset" / "processed"
TARGET_SIZE = (224, 224)
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
WINDOW_NAME = "Citrus - Manual Square Crop"
MAX_PREVIEW = (1100, 780)


def choose_square(image: Image.Image, label: str):
    """Return (x, y, side), 'skip', or 'quit'."""
    w, h = image.size
    scale = min(1.0, MAX_PREVIEW[0] / w, MAX_PREVIEW[1] / h)
    preview = cv2.cvtColor(
        np.asarray(image.resize((round(w * scale), round(h * scale)))),
        cv2.COLOR_RGB2BGR,
    )
    ph, pw = preview.shape[:2]
    min_side = min(max(32, round(min(w, h) * 0.15)), min(w, h))
    max_side = min(w, h)

    def initial_rect():
        side = min(max_side, max(min_side, round(max_side * 0.90)))
        return [(w - side) // 2, (h - side) // 2, side]

    state = {"rect": initial_rect(), "dragging": False, "offset": (0, 0)}

    def resize_square(multiplier: float):
        x, y, side = state["rect"]
        new_side = max(min_side, min(max_side, round(side * multiplier)))
        cx, cy = x + side / 2, y + side / 2
        nx = max(0, min(w - new_side, round(cx - new_side / 2)))
        ny = max(0, min(h - new_side, round(cy - new_side / 2)))
        state["rect"] = [nx, ny, new_side]

    def on_mouse(event, px, py, flags, userdata):
        ox = max(0, min(w - 1, int(px / scale)))
        oy = max(0, min(h - 1, int(py / scale)))
        x, y, side = state["rect"]

        if event == cv2.EVENT_LBUTTONDOWN:
            # Drag inside the square to move it. Click outside to reposition it.
            if not (x <= ox < x + side and y <= oy < y + side):
                x = max(0, min(w - side, ox - side // 2))
                y = max(0, min(h - side, oy - side // 2))
                state["rect"] = [x, y, side]
            state["offset"] = (ox - x, oy - y)
            state["dragging"] = True

        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            dx, dy = state["offset"]
            x = max(0, min(w - side, ox - dx))
            y = max(0, min(h - side, oy - dy))
            state["rect"] = [x, y, side]

        elif event == cv2.EVENT_LBUTTONUP:
            state["dragging"] = False

        elif event == cv2.EVENT_MOUSEWHEEL:
            # OpenCV Python does not expose getMouseWheelDelta on all builds.
            wheel_delta = (flags >> 16) & 0xFFFF
            if wheel_delta >= 32768:
                wheel_delta -= 65536
            if wheel_delta:
                resize_square(1.10 if wheel_delta > 0 else 1 / 1.10)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while True:
        x, y, side = state["rect"]
        x1, y1 = round(x * scale), round(y * scale)
        x2 = min(pw, round((x + side) * scale))
        y2 = min(ph, round((y + side) * scale))

        # Darken the excluded area and leave the selected square undimmed.
        canvas = cv2.convertScaleAbs(preview, alpha=0.55, beta=0)
        canvas[y1:y2, x1:x2] = preview[y1:y2, x1:x2]
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)
        cv2.putText(
            canvas, f"Square: {side} x {side}", (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
        )
        cv2.putText(
            canvas, "Drag: move | Wheel/+/-: size | Enter: save | N: skip | R: reset | Q: quit",
            (10, ph - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1,
        )
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):
            return tuple(state["rect"])
        if key == ord("n"):
            return "skip"
        if key in (ord("q"), 27):
            return "quit"
        if key == ord("r"):
            state["rect"] = initial_rect()
        if key in (ord("+"), ord("=")):
            resize_square(1.10)
        if key in (ord("-"), ord("_")):
            resize_square(1 / 1.10)
        # Closing the crop window ends the session instead of reopening it.
        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return "quit"
        except cv2.error:
            return "quit"


def main():
    if not RAW_DIR.is_dir():
        raise FileNotFoundError(f"Raw image folder not found: {RAW_DIR}")

    paths = sorted(
        p for p in RAW_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    print(f"Images found: {len(paths)}")
    saved = skipped = failed = 0

    try:
        for index, src_path in enumerate(paths, 1):
            relative = src_path.relative_to(RAW_DIR)
            print(f"[{index}/{len(paths)}] {relative}")
            try:
                with Image.open(src_path) as original:
                    image = ImageOps.exif_transpose(original).convert("RGB")
                result = choose_square(image, str(relative))
                if result == "quit":
                    break
                if result == "skip":
                    skipped += 1
                    continue
                x, y, side = result
                result_image = image.crop((x, y, x + side, y + side))
                result_image = result_image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                dest_path = (PROCESSED_DIR / relative).with_suffix(".png")
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                result_image.save(dest_path, format="PNG", optimize=True)
                saved += 1
                print(f"  Saved: {dest_path.relative_to(BASE_DIR)}")
            except Exception as exc:
                failed += 1
                print(f"  Failed: {exc}")
    finally:
        cv2.destroyAllWindows()

    print(f"Finished: saved={saved}, skipped={skipped}, failed={failed}")
    print("Original images in dataset/raw were not changed.")
    print("Output: 224x224, RGB, lossless PNG. Old JPG/JPEG outputs are not removed.")


if __name__ == "__main__":
    main()
