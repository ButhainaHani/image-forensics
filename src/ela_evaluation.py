
from pathlib import Path
import csv

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


# ==========================================
# 1. Project Configuration
# ==========================================
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(r"C:\Datasets\CASIA2")

IMAGE_DIR = DATASET_DIR / "Tp"
MASK_DIR = DATASET_DIR / "CASIA 2 Groundtruth"
ELA_DIR = PROJECT_DIR / "outputs" / "ela" / "ela_maps"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "ela"

MASK_THRESHOLD = 127
TEST_SIZE = 0.20
RANDOM_STATE = 42

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff"
}

# Thresholds will be selected using validation data only.
THRESHOLD_CANDIDATES = list(range(5, 256, 5))


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


def load_binary_mask(mask_path):
    with Image.open(mask_path) as img:
        mask = np.asarray(img.convert("L"))

    return mask > MASK_THRESHOLD


def load_ela_map(ela_path):
    with Image.open(ela_path) as img:
        ela = np.asarray(img.convert("RGB"), dtype=np.float32)

    # Convert RGB intensity into a single heuristic score.
    # This is not a probability of manipulation.
    return ela.mean(axis=2)


# ==========================================
# 3. Collect Available Samples
# ==========================================
def collect_samples():
    masks = load_masks()

    image_paths = sorted(
        path for path in IMAGE_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    samples = []
    skipped_rows = []

    for image_path in image_paths:
        mask_path = masks.get(image_path.stem)
        ela_path = ELA_DIR / f"{image_path.stem}_ela.png"

        if mask_path is None:
            skipped_rows.append({
                "image": image_path.name,
                "reason": "missing_mask",
            })
            continue

        if not ela_path.is_file():
            skipped_rows.append({
                "image": image_path.name,
                "reason": "missing_ela_map",
            })
            continue

        samples.append({
            "image": image_path.name,
            "mask_path": mask_path,
            "ela_path": ela_path,
        })

    return samples, skipped_rows, len(image_paths)


# ==========================================
# 4. Build Score Histograms
# ==========================================
def load_histograms(sample):
    ground_truth = load_binary_mask(sample["mask_path"])
    ela_score = load_ela_map(sample["ela_path"])

    if ground_truth.shape != ela_score.shape:
        raise ValueError("dimension_mismatch")

    if not ground_truth.any():
        raise ValueError("empty_ground_truth")

    # Integer bins preserve comparisons against integer thresholds.
    scores = np.clip(ela_score, 0, 255).astype(np.uint8)

    positive_hist = np.bincount(
        scores[ground_truth], minlength=256
    )
    negative_hist = np.bincount(
        scores[~ground_truth], minlength=256
    )

    return {
        "image": sample["image"],
        "positive_hist": positive_hist,
        "negative_hist": negative_hist,
        "total_positive": int(ground_truth.sum()),
    }


# ==========================================
# 5. Calculate Metrics From Pixel Counts
# ==========================================
def metrics_from_counts(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1_denominator = 2 * tp + fp + fn
    f1 = (
        (2 * tp) / f1_denominator
        if f1_denominator else 0.0
    )

    union = tp + fp + fn
    iou = tp / union if union else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def metrics_for_threshold(histogram, threshold):
    positive_hist = histogram["positive_hist"]
    negative_hist = histogram["negative_hist"]

    # Pixels at or above the threshold are predicted as tampered.
    tp = int(positive_hist[threshold:].sum())
    fp = int(negative_hist[threshold:].sum())
    fn = histogram["total_positive"] - tp

    return metrics_from_counts(tp, fp, fn)


# ==========================================
# 6. Choose Threshold Using Validation Data
# ==========================================
def choose_threshold(validation_samples, skipped_rows):
    validation_histograms = []

    print("\nPreparing validation data...")

    for index, sample in enumerate(validation_samples, start=1):
        try:
            histogram = load_histograms(sample)
            validation_histograms.append(histogram)

        except Exception as error:
            skipped_rows.append({
                "image": sample["image"],
                "reason": str(error),
            })

        if index % 250 == 0:
            print(
                f"Validation processed: "
                f"{index}/{len(validation_samples)}"
            )

    if not validation_histograms:
        raise RuntimeError(
            "No valid validation samples were available."
        )

    results = []

    for threshold in THRESHOLD_CANDIDATES:
        image_metrics = [
            metrics_for_threshold(histogram, threshold)
            for histogram in validation_histograms
        ]

        results.append({
            "threshold": threshold,
            "mean_precision": float(np.mean([
                item["precision"] for item in image_metrics
            ])),
            "mean_recall": float(np.mean([
                item["recall"] for item in image_metrics
            ])),
            "mean_f1": float(np.mean([
                item["f1"] for item in image_metrics
            ])),
            "mean_iou": float(np.mean([
                item["iou"] for item in image_metrics
            ])),
        })

    best = max(results, key=lambda row: row["mean_f1"])

    threshold_report = OUTPUT_DIR / "ela_threshold_validation.csv"

    with threshold_report.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "threshold",
                "mean_precision",
                "mean_recall",
                "mean_f1",
                "mean_iou",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    return best, threshold_report, len(validation_histograms)


# ==========================================
# 7. Evaluate Held-Out Test Data
# ==========================================
def evaluate_test_samples(test_samples, threshold, skipped_rows):
    test_rows = []
    valid_count = 0

    print("\nEvaluating held-out test images...")

    for index, sample in enumerate(test_samples, start=1):
        try:
            histogram = load_histograms(sample)
            metrics = metrics_for_threshold(histogram, threshold)

            test_rows.append({
                "image": sample["image"],
                "threshold": threshold,
                **metrics,
                "status": "evaluated",
            })
            valid_count += 1

        except Exception as error:
            skipped_rows.append({
                "image": sample["image"],
                "reason": str(error),
            })

        if index % 250 == 0:
            print(f"Test processed: {index}/{len(test_samples)}")

    return test_rows, valid_count


# ==========================================
# 8. Save CSV Reports
# ==========================================
def save_csv(path, rows, fields):
    with path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ==========================================
# 9. Main
# ==========================================
def main():
    for folder in (IMAGE_DIR, MASK_DIR, ELA_DIR):
        if not folder.exists():
            raise FileNotFoundError(
                f"Required folder not found: {folder}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading image, mask, and ELA file paths...")

    samples, skipped_rows, total_images = collect_samples()

    if len(samples) < 10:
        raise RuntimeError(
            f"Only {len(samples)} samples found. "
            "At least 10 are required."
        )

    # Keep a separate held-out test set.
    validation_samples, test_samples = train_test_split(
        samples,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    print("Available samples:", len(samples))
    print("Validation candidates:", len(validation_samples))
    print("Test candidates:", len(test_samples))

    best, threshold_report, valid_validation_count = (
        choose_threshold(validation_samples, skipped_rows)
    )

    test_rows, valid_test_count = evaluate_test_samples(
        test_samples,
        best["threshold"],
        skipped_rows,
    )

    evaluation_path = OUTPUT_DIR / "ela_evaluation.csv"
    save_csv(
        evaluation_path,
        test_rows,
        [
            "image",
            "threshold",
            "precision",
            "recall",
            "f1",
            "iou",
            "status",
        ],
    )

    skipped_path = OUTPUT_DIR / "ela_skipped_images.csv"
    save_csv(
        skipped_path,
        skipped_rows,
        ["image", "reason"],
    )

    print("\n========== ELA EVALUATION SUMMARY ==========")
    print("Images found:", total_images)
    print("Samples with matching files:", len(samples))
    print("Valid validation images:", valid_validation_count)
    print("Valid test images:", valid_test_count)
    print("Skipped/problematic entries:", len(skipped_rows))

    print("\n========== VALIDATION RESULTS ==========")
    print("Best threshold:", best["threshold"])
    print(f"Validation mean Precision: {best['mean_precision']:.4f}")
    print(f"Validation mean Recall:    {best['mean_recall']:.4f}")
    print(f"Validation mean F1:        {best['mean_f1']:.4f}")
    print(f"Validation mean IoU:       {best['mean_iou']:.4f}")

    print("\n========== HELD-OUT TEST RESULTS ==========")

    if test_rows:
        for metric in ("precision", "recall", "f1", "iou"):
            values = [row[metric] for row in test_rows]
            print(f"Mean {metric}: {np.mean(values):.4f}")
    else:
        print("No valid test images were evaluated.")

    print("\nReports saved:")
    print("Evaluation:", evaluation_path)
    print("Threshold search:", threshold_report)
    print("Skipped images:", skipped_path)
    print("\nELA evaluation finished.")


if __name__ == "__main__":
    main()
