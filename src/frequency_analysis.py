
from pathlib import Path
import cv2
import numpy as np
import csv

# ==========================================
# 1. Project Configuration & Paths
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(r"C:\Datasets\CASIA2")

TP_FOLDER = DATASET_DIR / "Tp"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "frequency"
DCT_MAP_DIR = OUTPUT_DIR / "frequency_maps"
FFT_MAP_DIR = OUTPUT_DIR / "fft_maps"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DCT_MAP_DIR.mkdir(parents=True, exist_ok=True)
FFT_MAP_DIR.mkdir(parents=True, exist_ok=True)

print("Frequency Analysis environment is ready!", flush=True)


# ==========================================
# 2. DCT Analysis
# ==========================================

def perform_dct_analysis(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_float = np.float32(gray) / 255.0

    dct_image = cv2.dct(gray_float)
    dct_log = np.log(np.abs(dct_image) + 1e-5)

    dct_log[0, 0] = 0

    frequency_map = cv2.normalize(
        dct_log, None, 0, 255, cv2.NORM_MINMAX
    )

    return frequency_map.astype(np.uint8), dct_log


# ==========================================
# 3. FFT Analysis
# ==========================================

def perform_fft_analysis(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_float = np.float32(gray)

    fft_image = np.fft.fft2(gray_float)
    fft_shift = np.fft.fftshift(fft_image)

    magnitude_spectrum = np.abs(fft_shift)
    fft_log = np.log1p(magnitude_spectrum)

    center_y = fft_log.shape[0] // 2
    center_x = fft_log.shape[1] // 2
    fft_log[center_y, center_x] = 0

    fft_map = cv2.normalize(
        fft_log, None, 0, 255, cv2.NORM_MINMAX
    )

    return fft_map.astype(np.uint8), fft_log


# ==========================================
# 4. Feature Extraction
# ==========================================

def extract_frequency_features(transform, prefix):

    height, width = transform.shape

    y, x = np.ogrid[:height, :width]

    center_y = height // 2
    center_x = width // 2

    if prefix == "dct":
        # DCT low frequencies are located at the top-left
        radius = np.sqrt(
            (y / max(height, 1)) ** 2 +
            (x / max(width, 1)) ** 2
        )
    else:
        # FFT low frequencies are located at the center
        radius = np.sqrt(
            ((y - center_y) / max(height, 1)) ** 2 +
            ((x - center_x) / max(width, 1)) ** 2
        )

    high_frequency_mask = radius > 0.25

    absolute_values = np.abs(transform)
    total_energy = np.sum(absolute_values ** 2) + 1e-10

    high_frequency_energy = np.sum(
        absolute_values[high_frequency_mask] ** 2
    )

    features = {
        f"{prefix}_mean": float(np.mean(transform)),
        f"{prefix}_std": float(np.std(transform)),
        f"{prefix}_median": float(np.median(transform)),
        f"{prefix}_p90": float(np.percentile(transform, 90)),
        f"{prefix}_high_frequency_energy_ratio": float(
            high_frequency_energy / total_energy
        )
    }

    return features


# ==========================================
# 5. Feature Normalization
# ==========================================

def normalize_features(rows, feature_names):

    for feature_name in feature_names:

        values = np.array(
            [row[feature_name] for row in rows],
            dtype=np.float64
        )

        minimum = np.min(values)
        maximum = np.max(values)

        if maximum == minimum:
            normalized_values = np.zeros_like(values)
        else:
            normalized_values = (
                (values - minimum) / (maximum - minimum)
            )

        normalized_name = feature_name + "_normalized"

        for row, normalized_value in zip(
            rows, normalized_values
        ):
            row[normalized_name] = float(normalized_value)


# ==========================================
# 6. Main Processing Function
# ==========================================

def main():

    image_extensions = {
        ".jpg", ".jpeg", ".png",
        ".bmp", ".tif", ".tiff"
    }

    if not TP_FOLDER.exists():
        raise FileNotFoundError(
            f"Image folder not found: {TP_FOLDER}"
        )

    image_paths = sorted(
        p for p in TP_FOLDER.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    )

    print(
        f"Starting DCT and FFT Analysis on {len(image_paths)} images..."
    )

    report_rows = []
    feature_rows = []

    dct_success_count = 0
    fft_success_count = 0

    for index, image_path in enumerate(image_paths, start=1):

        try:

            image = cv2.imread(str(image_path))

            if image is None:
                raise ValueError(
                    f"Could not read image: {image_path}"
                )

            # DCT
            dct_map, dct_transform = perform_dct_analysis(image)

            dct_output_path = (
                DCT_MAP_DIR / (image_path.stem + "_freq.png")
            )

            if not cv2.imwrite(str(dct_output_path), dct_map):
                raise IOError("Could not save DCT map")

            dct_success_count += 1

            # FFT
            fft_map, fft_transform = perform_fft_analysis(image)

            fft_output_path = (
                FFT_MAP_DIR / (image_path.stem + "_fft.png")
            )

            if not cv2.imwrite(str(fft_output_path), fft_map):
                raise IOError("Could not save FFT map")

            fft_success_count += 1

            # Extract numerical features
            features = {"image": image_path.name}

            features.update(
                extract_frequency_features(
                    dct_transform, "dct"
                )
            )

            features.update(
                extract_frequency_features(
                    fft_transform, "fft"
                )
            )

            feature_rows.append(features)

            report_rows.append({
                "image": image_path.name,
                "dct_map": str(dct_output_path),
                "fft_map": str(fft_output_path),
                "status": "success"
            })

        except Exception as error:

            report_rows.append({
                "image": image_path.name,
                "dct_map": "",
                "fft_map": "",
                "status": f"error: {error}"
            })

        if index % 250 == 0:
            print(
                f"Processed {index}/{len(image_paths)} images"
            )

    # ==========================================
    # 7. Normalize Extracted Features
    # ==========================================

    feature_names = [
        key for key in feature_rows[0].keys()
        if key != "image"
    ] if feature_rows else []

    if feature_rows:
        normalize_features(feature_rows, feature_names)

    # ==========================================
    # 8. Save Frequency Analysis Report
    # ==========================================

    report_path = OUTPUT_DIR / "frequency_report.csv"

    with report_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image", "dct_map", "fft_map", "status"
            ]
        )

        writer.writeheader()
        writer.writerows(report_rows)

    # ==========================================
    # 9. Save Extracted and Normalized Features
    # ==========================================

    features_path = OUTPUT_DIR / "frequency_features.csv"

    if feature_rows:

        feature_fieldnames = list(feature_rows[0].keys())

        normalized_names = [
            name + "_normalized" for name in feature_names
        ]

        feature_fieldnames.extend(normalized_names)

        with features_path.open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:

            writer = csv.DictWriter(
                file, fieldnames=feature_fieldnames
            )

            writer.writeheader()
            writer.writerows(feature_rows)

    # ==========================================
    # 10. Final Summary
    # ==========================================

    print("\n========== FREQUENCY ANALYSIS SUMMARY ==========")
    print("Total images processed:", len(image_paths))
    print("Successful DCT maps:", dct_success_count)
    print("Successful FFT maps:", fft_success_count)
    print("Images with extracted features:", len(feature_rows))
    print("DCT output folder:", DCT_MAP_DIR)
    print("FFT output folder:", FFT_MAP_DIR)
    print("Analysis report:", report_path)
    print("Features report:", features_path)
    print("Frequency analysis finished successfully.")


if __name__ == "__main__":
    main()
