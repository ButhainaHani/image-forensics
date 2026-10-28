
from pathlib import Path
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import io
import csv

# ==========================================
# 1. Project Configuration
# ==========================================
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(r"C:\Datasets\CASIA2")

IMAGE_DIR = DATASET_DIR / "Tp"
MASK_DIR = DATASET_DIR / "CASIA 2 Groundtruth"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "ela"
MAP_DIR = OUTPUT_DIR / "ela_maps"
COMPARISON_DIR = OUTPUT_DIR / "comparisons"

JPEG_QUALITY = 90
ELA_SCALE = 15
SAVE_COMPARISONS = 20

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff"
}

for folder in (MAP_DIR, COMPARISON_DIR):
    folder.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. Load Ground Truth Masks
# ==========================================
def load_masks():
    masks = {}

    for path in MASK_DIR.rglob("*"):
        if path.is_file() and path.stem.endswith("_gt"):
            image_stem = path.stem.removesuffix("_gt")
            masks[image_stem] = path

    return masks


# ==========================================
# 3. ELA Algorithm
# ==========================================
def perform_ela(image_path):
    with Image.open(image_path) as img:
        original = img.convert("RGB")

    # Recompress image as JPEG in memory
    buffer = io.BytesIO()
    original.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    buffer.seek(0)

    with Image.open(buffer) as compressed:
        compressed = compressed.convert("RGB")

    # Calculate absolute difference
    difference = ImageChops.difference(
        original, compressed
    )

    # Amplify differences for visualization
    ela_image = ImageEnhance.Brightness(
        difference
    ).enhance(ELA_SCALE)

    # Calculate mean error before amplification
    diff_array = np.asarray(difference, dtype=np.float32)
    mean_error = float(diff_array.mean())

    return original, ela_image, mean_error


# ==========================================
# 4. Save Comparison Image
# ==========================================
def save_comparison(original, ela_image, mask_path, output_path):
    with Image.open(mask_path) as mask:
        mask = mask.convert("L")

    # White pixels represent annotated tampered regions
    mask_rgb = mask.convert("RGB")

    width, height = original.size
    canvas = Image.new("RGB", (width * 3, height), "white")

    canvas.paste(original, (0, 0))
    canvas.paste(ela_image, (width, 0))
    canvas.paste(mask_rgb, (width * 2, 0))

    canvas.save(output_path)


# ==========================================
# 5. Run ELA on CASIA2
# ==========================================
def main():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image folder not found: {IMAGE_DIR}"
        )

    if not MASK_DIR.exists():
        raise FileNotFoundError(
            f"Ground truth folder not found: {MASK_DIR}"
        )

    masks = load_masks()

    image_paths = sorted(
        p for p in IMAGE_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    print("Starting ELA analysis...")
    print("Images found:", len(image_paths))
    print("Ground truth masks found:", len(masks))

    report_rows = []
    success_count = 0
    comparison_count = 0

    for index, image_path in enumerate(image_paths, start=1):
        mask_path = masks.get(image_path.stem)

        if mask_path is None:
            report_rows.append({
                "image": image_path.name,
                "mask": "",
                "mean_error": "",
                "ela_map": "",
                "status": "skipped_missing_mask"
            })
            continue

        try:
            with Image.open(mask_path) as mask_img:
                mask_size = mask_img.size

            with Image.open(image_path) as img:
                image_size = img.size

            if image_size != mask_size:
                report_rows.append({
                    "image": image_path.name,
                    "mask": mask_path.name,
                    "mean_error": "",
                    "ela_map": "",
                    "status": "skipped_dimension_mismatch"
                })
                continue

            original, ela_image, mean_error = perform_ela(
                image_path
            )

            output_name = image_path.stem + "_ela.png"
            ela_path = MAP_DIR / output_name
            ela_image.save(ela_path)

            if comparison_count < SAVE_COMPARISONS:
                comparison_path = (
                    COMPARISON_DIR /
                    f"{image_path.stem}_comparison.png"
                )

                save_comparison(
                    original,
                    ela_image,
                    mask_path,
                    comparison_path
                )

                comparison_count += 1

            report_rows.append({
                "image": image_path.name,
                "mask": mask_path.name,
                "mean_error": mean_error,
                "ela_map": str(ela_path),
                "status": "success"
            })

            success_count += 1

        except Exception as error:
            report_rows.append({
                "image": image_path.name,
                "mask": mask_path.name,
                "mean_error": "",
                "ela_map": "",
                "status": f"error: {error}"
            })

        if index % 250 == 0:
            print(f"Scanned {index}/{len(image_paths)} images")

    # Save CSV report
    report_path = OUTPUT_DIR / "ela_report.csv"

    with report_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image", "mask", "mean_error",
                "ela_map", "status"
            ]
        )
        writer.writeheader()
        writer.writerows(report_rows)

    print("\n========== ELA SUMMARY ==========")
    print("Images scanned:", len(image_paths))
    print("Successful ELA maps:", success_count)
    print("Comparison images saved:", comparison_count)
    print("Output folder:", OUTPUT_DIR)
    print("Report:", report_path)
    print("ELA processing finished.")


if __name__ == "__main__":
    main()
