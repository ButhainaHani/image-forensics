
from pathlib import Path
import cv2
import numpy as np
import csv

# ==========================================
# PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

ELA_DIR = OUTPUT_DIR / "ela" / "ela_maps"
NOISE_DIR = OUTPUT_DIR / "noise" / "noise_maps"

DATASET_DIR = Path(r"C:\Datasets\CASIA2")
TP_DIR = DATASET_DIR / "Tp"
GT_DIR = DATASET_DIR / "CASIA 2 Groundtruth"

EXPERIMENT_DIR = OUTPUT_DIR / "phase5_experiments"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# LOAD MAPS
# ==========================================

def load_maps(folder, suffix):
    result = {}

    if not folder.exists():
        print(f"Warning: Folder not found: {folder}")
        return result

    for path in folder.glob(f"*_{suffix}.png"):
        name = path.stem[:-len(suffix)-1]

        image = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is not None:
            result[name] = image

    return result


def find_original_image(name):
    matches = sorted(TP_DIR.glob(name + ".*"))

    for path in matches:
        image = cv2.imread(str(path))

        if image is not None:
            return image

    return None


# ==========================================
# ROBUST NORMALIZATION
# ==========================================

def normalize_map(image, size):
    """
    Resize and normalize a map to [0, 1].
    Percentile clipping reduces the effect of extreme values.
    """

    image = cv2.resize(
        image,
        size,
        interpolation=cv2.INTER_LINEAR
    )

    image = image.astype(np.float32)

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    lower = np.percentile(image, 1)
    upper = np.percentile(image, 99)

    if upper <= lower:
        return np.zeros(image.shape, dtype=np.float32)

    image = np.clip(image, lower, upper)

    image = (image - lower) / (upper - lower)

    return image.astype(np.float32)


# ==========================================
# SPATIAL HIGH-FREQUENCY ANALYSIS
# ==========================================

def create_spatial_frequency_map(image, size):
    """
    Generate a spatial high-frequency energy map
    using the Laplacian operator.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    high_frequency = cv2.Laplacian(
        gray,
        cv2.CV_32F,
        ksize=3
    )

    energy = np.abs(high_frequency)

    energy = cv2.GaussianBlur(
        energy,
        (9, 9),
        0
    )

    return normalize_map(energy, size)


# ==========================================
# METRICS
# ==========================================

def calculate_metrics(prediction, ground_truth):

    pred = prediction > 0
    gt = ground_truth > 0

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    f1 = (
        2 * precision * recall
        / (precision + recall + 1e-8)
    )

    iou = tp / (tp + fp + fn + 1e-8)

    return precision, recall, f1, iou


# ==========================================
# COMBINATION AND PREDICTION
# ==========================================

def make_prediction(images):

    combined = np.mean(
        images,
        axis=0
    )

    combined = np.clip(
        combined * 255,
        0,
        255
    ).astype(np.uint8)

    _, binary = cv2.threshold(
        combined,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return combined, binary


# ==========================================
# MAIN
# ==========================================

def main():

    print("Loading forensic maps...")

    ela = load_maps(ELA_DIR, "ela")
    noise = load_maps(NOISE_DIR, "noise")

    print(f"ELA maps: {len(ela)}")
    print(f"Noise maps: {len(noise)}")

    common = set(ela) & set(noise)

    print(f"Common images: {len(common)}")

    rows = []

    processed = 0
    skipped = 0

    experiment_names = [
        "ELA",
        "Noise",
        "Frequency",
        "ELA + Noise",
        "ELA + Noise + Frequency"
    ]

    # ======================================
    # PROCESS IMAGES
    # ======================================

    for index, name in enumerate(sorted(common), 1):

        gt_path = GT_DIR / f"{name}_gt.png"

        if not gt_path.exists():
            skipped += 1
            continue

        gt = cv2.imread(
            str(gt_path),
            cv2.IMREAD_GRAYSCALE
        )

        original = find_original_image(name)

        if gt is None or original is None:
            skipped += 1
            continue

        height, width = gt.shape
        size = (width, height)

        gt_binary = (
            (gt > 0).astype(np.uint8) * 255
        )

        # Normalize existing spatial maps
        ela_map = normalize_map(
            ela[name],
            size
        )

        noise_map = normalize_map(
            noise[name],
            size
        )

        # Generate spatial frequency map
        frequency_map = create_spatial_frequency_map(
            original,
            size
        )

        maps = {
            "ELA": ela_map,
            "Noise": noise_map,
            "Frequency": frequency_map
        }

        experiments = {
            "ELA": [maps["ELA"]],

            "Noise": [maps["Noise"]],

            "Frequency": [maps["Frequency"]],

            "ELA + Noise": [
                maps["ELA"],
                maps["Noise"]
            ],

            "ELA + Noise + Frequency": [
                maps["ELA"],
                maps["Noise"],
                maps["Frequency"]
            ]
        }

        # ==================================
        # RUN EXPERIMENTS
        # ==================================

        for experiment_name in experiment_names:

            inputs = experiments[experiment_name]

            combined, binary = make_prediction(inputs)

            safe_name = (
                experiment_name
                .lower()
                .replace(" + ", "_")
                .replace(" ", "_")
            )

            output_folder = EXPERIMENT_DIR / safe_name
            output_folder.mkdir(
                parents=True,
                exist_ok=True
            )

            output_path = (
                output_folder / f"{name}_map.png"
            )

            cv2.imwrite(
                str(output_path),
                combined
            )

            precision, recall, f1, iou = calculate_metrics(
                binary,
                gt_binary
            )

            rows.append({
                "image": name,
                "experiment": experiment_name,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "iou": iou,
                "evaluation_type": "pixel_metrics"
            })

        processed += 1

        if index % 250 == 0:
            print(
                f"Processed {index}/{len(common)} images"
            )

    # ======================================
    # SAVE REPORT
    # ======================================

    report_path = (
        EXPERIMENT_DIR / "experiment_comparison.csv"
    )

    fields = [
        "image",
        "experiment",
        "precision",
        "recall",
        "f1_score",
        "iou",
        "evaluation_type"
    ]

    with open(
        report_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(rows)

    # ======================================
    # SUMMARY
    # ======================================

    print("\n========== PHASE 5 EXPERIMENT SUMMARY ==========")

    print(f"Images processed: {processed}")
    print(f"Images skipped: {skipped}")
    print(f"Experiment results: {len(rows)}")

    print(f"Report: {report_path}")
    print(f"Output folder: {EXPERIMENT_DIR}")

    print("\nNormalization: Percentile clipping (1st-99th)")
    print("Frequency method: Spatial Laplacian energy")
    print("Experiments finished successfully.")


if __name__ == "__main__":
    main()
