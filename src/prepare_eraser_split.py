from pathlib import Path
import cv2
import numpy as np

# ==========================================
# 1. PATHS & CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
ERASER_RAW_DIR = Path(
    r"D:\Amira\Uni\level 3\semester 5\CSE281 Image Processing\CSE281 Project\Datasets\eraser_benchmark_results"
)
ERASER_OUT_DIR = PROJECT_DIR / "outputs" / "eraser_prepared"
ORIG_OUT = ERASER_OUT_DIR / "original"
MASK_OUT = ERASER_OUT_DIR / "groundtruth"
REMOVED_OUT = ERASER_OUT_DIR / "tampered_removed"

for d in (ORIG_OUT, MASK_OUT, REMOVED_OUT):
    d.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ==========================================
# 2. SPLIT TRIPTYCH DEMO IMAGES
# ==========================================

def main():
    if not ERASER_RAW_DIR.exists():
        raise FileNotFoundError(f"Eraser folder not found: {ERASER_RAW_DIR}")

    files = sorted(
        p for p in ERASER_RAW_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    processed = 0
    skipped = 0

    for path in files:
        img = cv2.imread(str(path))
        if img is None:
            skipped += 1
            continue

        h, w = img.shape[:2]
        third = w // 3
        if third == 0:
            skipped += 1
            continue

        orig_part = img[:, :third]
        mask_part = img[:, third : 2 * third]
        removed_part = img[:, 2 * third : 3 * third]

        # Convert mask part to clean binary mask (0 or 255)
        mask_gray = cv2.cvtColor(mask_part, cv2.COLOR_BGR2GRAY)
        mask_bin = ((mask_gray > 127).astype(np.uint8)) * 255

        stem = path.stem
        cv2.imwrite(str(ORIG_OUT / f"{stem}_orig.png"), orig_part)
        cv2.imwrite(str(MASK_OUT / f"{stem}_gt.png"), mask_bin)
        cv2.imwrite(str(REMOVED_OUT / f"{stem}_removed.png"), removed_part)
        processed += 1

    print(f"Processed: {processed} images | Skipped: {skipped}")
    print(f"Saved to: {ERASER_OUT_DIR}")


if __name__ == "__main__":
    main()