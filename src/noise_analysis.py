
from pathlib import Path
import cv2
import numpy as np
import csv
from scipy.ndimage import uniform_filter


# =========================
# PATHS
# =========================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = Path(r"C:\Datasets\CASIA2")
TP_FOLDER = DATASET_DIR / "Tp"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "noise"
NOISE_MAP_DIR = OUTPUT_DIR / "noise_maps"
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"

REPORT_PATH = OUTPUT_DIR / "noise_report.csv"


# =========================
# NOISE ANALYSIS
# =========================

def extract_noise_residual(image, ksize=3):
    """Extract high-frequency noise residual."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_float = gray.astype(np.float32)

    smoothed = cv2.medianBlur(gray, ksize).astype(np.float32)

    residual = gray_float - smoothed

    return residual


def compute_local_noise_map(residual, window_size=7):
    """Calculate local noise variance and enhance it."""

    mean = uniform_filter(residual, size=window_size)
    mean_sq = uniform_filter(residual ** 2, size=window_size)

    variance = np.maximum(mean_sq - mean ** 2, 0)

    log_variance = np.log1p(variance)

    norm_map = cv2.normalize(
        log_variance,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    norm_map = norm_map.astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_map = clahe.apply(norm_map)

    return enhanced_map


def generate_noise_heatmap(noise_map):
    """Generate colored noise heatmap."""

    return cv2.applyColorMap(
        noise_map,
        cv2.COLORMAP_JET
    )


# =========================
# PROCESS DATASET
# =========================

def main():

    NOISE_MAP_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

    extensions = [
        "*.jpg", "*.jpeg", "*.png",
        "*.tif", "*.tiff", "*.bmp"
    ]

    image_paths = []

    for ext in extensions:
        image_paths.extend(TP_FOLDER.glob(ext))

    image_paths = sorted(image_paths)

    print(f"Found {len(image_paths)} images.")

    report_rows = []

    success_count = 0
    error_count = 0

    for idx, img_path in enumerate(image_paths, 1):

        try:
            image = cv2.imread(str(img_path))

            if image is None:
                raise ValueError("Could not read image")

            residual = extract_noise_residual(image)

            noise_map = compute_local_noise_map(residual)

            heatmap = generate_noise_heatmap(noise_map)

            noise_path = NOISE_MAP_DIR / f"{img_path.stem}_noise.png"
            heatmap_path = HEATMAP_DIR / f"{img_path.stem}_heatmap.png"

            noise_saved = cv2.imwrite(str(noise_path), noise_map)
            heatmap_saved = cv2.imwrite(str(heatmap_path), heatmap)

            if not noise_saved or not heatmap_saved:
                raise IOError("Could not save output maps")

            success_count += 1

            report_rows.append({
                "image": img_path.name,
                "noise_map": noise_path.name,
                "heatmap": heatmap_path.name,
                "status": "success"
            })

        except Exception as e:

            error_count += 1

            report_rows.append({
                "image": img_path.name,
                "noise_map": "",
                "heatmap": "",
                "status": f"error: {e}"
            })

        if idx % 250 == 0:
            print(f"Processed {idx}/{len(image_paths)} images")

    # Save report

    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f:

        fieldnames = [
            "image",
            "noise_map",
            "heatmap",
            "status"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(report_rows)

    print("\n========== NOISE ANALYSIS SUMMARY ==========")
    print(f"Total images: {len(image_paths)}")
    print(f"Successful noise maps: {success_count}")
    print(f"Failed images: {error_count}")
    print(f"Noise maps folder: {NOISE_MAP_DIR}")
    print(f"Heatmaps folder: {HEATMAP_DIR}")
    print(f"Report: {REPORT_PATH}")
    print("Noise analysis finished.")


if __name__ == "__main__":
    main()
