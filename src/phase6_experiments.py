from pathlib import Path
import csv
import random
import cv2
import numpy as np

# ==========================================
# 1. PATHS & CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

ELA_DIR = OUTPUT_DIR / "ela" / "ela_maps"
NOISE_DIR = OUTPUT_DIR / "noise" / "noise_maps"
DCT_DIR = OUTPUT_DIR / "frequency" / "frequency_maps"
FFT_DIR = OUTPUT_DIR / "frequency" / "fft_maps"

DATASET_DIR = Path(
    r"D:\Amira\Uni\level 3\semester 5\CSE281 Image Processing\CSE281 Project\Datasets\CASIA2"
)
TP_DIR = DATASET_DIR / "Tp"
GT_DIR = DATASET_DIR / "CASIA 2 Groundtruth"

PHASE6_DIR = OUTPUT_DIR / "phase6_experiments"
PHASE6_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
TEST_RATIO = 0.20
RANDOM_SEED = 42


# ==========================================
# 2. FILE LOADING HELPERS
# ==========================================

def load_maps(folder, suffix):
    """Load grayscale maps keyed by image base stem."""
    maps = {}
    if not folder.exists():
        print(f"Warning: Folder not found -> {folder}")
        return maps

    for path in folder.glob(f"*_{suffix}.png"):
        stem = path.stem[:-(len(suffix) + 1)]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            maps[stem] = image

    return maps


def load_masks(gt_folder):
    """Index ground-truth masks by base image stem (stripping _gt)."""
    masks = {}
    if not gt_folder.exists():
        print(f"Warning: GT folder not found -> {gt_folder}")
        return masks

    for path in gt_folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem.lower().endswith("_gt"):
                stem = path.stem[:-3]
                masks.setdefault(stem, path)

    return masks


def find_original_image(stem, tp_folder):
    """Find matching original tampered image across supported extensions."""
    for ext in IMAGE_EXTENSIONS:
        for candidate in (tp_folder / f"{stem}{ext}", tp_folder / f"{stem}{ext.upper()}"):
            if candidate.is_file():
                img = cv2.imread(str(candidate))
                if img is not None:
                    return img
    return None


# ==========================================
# 3. NORMALIZATION & PREDICTION
# ==========================================

def normalize_map(image, size):
    """Resize and normalize map to [0, 1] using 1st-99th percentile clipping."""
    resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    lower = np.percentile(arr, 1)
    upper = np.percentile(arr, 99)

    if upper <= lower:
        return np.zeros(arr.shape, dtype=np.float32)

    clipped = np.clip(arr, lower, upper)
    return ((clipped - lower) / (upper - lower)).astype(np.float32)


def make_prediction(maps_list):
    """Combine maps via mean and apply Otsu thresholding."""
    combined_float = np.mean(maps_list, axis=0)
    combined_u8 = np.clip(combined_float * 255.0, 0, 255).astype(np.uint8)

    _, binary_mask = cv2.threshold(
        combined_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return combined_u8, binary_mask


# ==========================================
# 4. ALL 6 PROPOSAL METRICS
# ==========================================

def compute_roc_auc_u8(score_u8, gt_bool):
    """Fast exact ROC AUC computation on uint8 score maps [0..255]."""
    pos_count = int(gt_bool.sum())
    neg_count = int((~gt_bool).sum())

    if pos_count == 0 or neg_count == 0:
        return 0.5

    pos_hist = np.bincount(score_u8[gt_bool], minlength=256).astype(np.float64)
    neg_hist = np.bincount(score_u8[~gt_bool], minlength=256).astype(np.float64)

    # Sort thresholds descending (from 255 down to 0)
    tps = np.cumsum(pos_hist[::-1])
    fps = np.cumsum(neg_hist[::-1])

    tpr = np.concatenate(([0.0], tps / pos_count))
    fpr = np.concatenate(([0.0], fps / neg_count))

    # Trapezoidal integration
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) * 0.5))
    return auc


def calculate_all_metrics(score_u8, binary_pred, ground_truth):
    """Compute Precision, Recall, F1-score, IoU, Accuracy, and AUC."""
    pred = binary_pred > 0
    gt = ground_truth > 0

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())

    eps = 1e-8
    precision = float(tp / (tp + fp + eps))
    recall = float(tp / (tp + fn + eps))
    f1 = float((2.0 * precision * recall) / (precision + recall + eps))
    iou = float(tp / (tp + fp + fn + eps))
    accuracy = float((tp + tn) / (tp + tn + fp + fn + eps))
    auc = compute_roc_auc_u8(score_u8, gt)

    return precision, recall, f1, iou, accuracy, auc


# ==========================================
# 5. AGGREGATION & SAVING HELPERS
# ==========================================

def summarize_rows(rows, experiment_order):
    summary = []
    for exp_name in experiment_order:
        exp_rows = [r for r in rows if r["experiment"] == exp_name]
        if not exp_rows:
            continue

        summary.append({
            "experiment": exp_name,
            "samples_count": len(exp_rows),
            "mean_precision": float(np.mean([r["precision"] for r in exp_rows])),
            "mean_recall": float(np.mean([r["recall"] for r in exp_rows])),
            "mean_f1_score": float(np.mean([r["f1_score"] for r in exp_rows])),
            "mean_iou": float(np.mean([r["iou"] for r in exp_rows])),
            "mean_accuracy": float(np.mean([r["accuracy"] for r in exp_rows])),
            "mean_auc": float(np.mean([r["auc"] for r in exp_rows])),
        })

    summary.sort(key=lambda r: r["mean_f1_score"], reverse=True)
    return summary


def save_summary_csv(path, summary_rows):
    fields = [
        "experiment",
        "samples_count",
        "mean_precision",
        "mean_recall",
        "mean_f1_score",
        "mean_iou",
        "mean_accuracy",
        "mean_auc",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)


def print_summary_table(title, summary_rows):
    print(f"\n========== {title} ==========")
    print(
        f"{'Experiment':<26} {'Prec':>8} {'Recall':>8} {'F1':>8} "
        f"{'IoU':>8} {'Acc':>8} {'AUC':>8}"
    )
    for r in summary_rows:
        print(
            f"{r['experiment']:<26} "
            f"{r['mean_precision']:>8.4f} "
            f"{r['mean_recall']:>8.4f} "
            f"{r['mean_f1_score']:>8.4f} "
            f"{r['mean_iou']:>8.4f} "
            f"{r['mean_accuracy']:>8.4f} "
            f"{r['mean_auc']:>8.4f}"
        )


# ==========================================
# 6. MAIN PHASE 6 PIPELINE
# ==========================================

def main():
    print("Loading ELA, Noise, DCT, and FFT maps...")

    ela_maps = load_maps(ELA_DIR, "ela")
    noise_maps = load_maps(NOISE_DIR, "noise")
    dct_maps = load_maps(DCT_DIR, "freq")
    fft_maps = load_maps(FFT_DIR, "fft")
    gt_masks = load_masks(GT_DIR)

    common_stems = sorted(
        set(ela_maps) & set(noise_maps) & set(dct_maps) & set(fft_maps) & set(gt_masks)
    )
    print(f"Complete aligned samples (with GT): {len(common_stems)}")

    if not common_stems:
        raise RuntimeError("No common samples found across all required maps and GT masks.")

    # Deterministic 80/20 Validation / Held-Out Test split
    shuffled = list(common_stems)
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1.0 - TEST_RATIO))
    val_set = set(shuffled[:split_idx])
    test_set = set(shuffled[split_idx:])

    print(f"Validation split: {len(val_set)} | Held-out Test split: {len(test_set)}")

    experiment_order = [
        "ELA",
        "Noise",
        "Frequency",
        "ELA + Noise",
        "ELA + Noise + Frequency",
        "DCT only",
        "FFT only",
    ]

    detailed_rows = []
    processed = 0
    skipped = 0

    for idx, stem in enumerate(common_stems, start=1):
        gt_img = cv2.imread(str(gt_masks[stem]), cv2.IMREAD_GRAYSCALE)
        orig_img = find_original_image(stem, TP_DIR)

        if gt_img is None or orig_img is None:
            skipped += 1
            continue

        height, width = gt_img.shape[:2]
        size = (width, height)
        gt_binary = ((gt_img > 127).astype(np.uint8)) * 255
        split_label = "test" if stem in test_set else "validation"

        ela_norm = normalize_map(ela_maps[stem], size)
        noise_norm = normalize_map(noise_maps[stem], size)
        dct_norm = normalize_map(dct_maps[stem], size)
        fft_norm = normalize_map(fft_maps[stem], size)
        freq_norm = ((dct_norm + fft_norm) / 2.0).astype(np.float32)

        experiments = {
            "ELA": [ela_norm],
            "Noise": [noise_norm],
            "Frequency": [freq_norm],
            "ELA + Noise": [ela_norm, noise_norm],
            "ELA + Noise + Frequency": [ela_norm, noise_norm, freq_norm],
            "DCT only": [dct_norm],
            "FFT only": [fft_norm],
        }

        for exp_name in experiment_order:
            combined_u8, binary_pred = make_prediction(experiments[exp_name])

            safe_folder_name = (
                exp_name.lower().replace(" + ", "_").replace(" ", "_")
            )
            exp_out_dir = PHASE6_DIR / safe_folder_name
            exp_out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(exp_out_dir / f"{stem}_map.png"), combined_u8)

            precision, recall, f1, iou, accuracy, auc = calculate_all_metrics(
                combined_u8, binary_pred, gt_binary
            )

            detailed_rows.append({
                "image": stem,
                "split": split_label,
                "experiment": exp_name,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "iou": iou,
                "accuracy": accuracy,
                "auc": auc,
            })

        processed += 1
        if idx % 500 == 0:
            print(f"Processed {idx}/{len(common_stems)} images...")

    # Save detailed per-image CSV
    detailed_csv_path = PHASE6_DIR / "phase6_detailed_results.csv"
    with detailed_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "split",
                "experiment",
                "precision",
                "recall",
                "f1_score",
                "iou",
                "accuracy",
                "auc",
            ],
        )
        writer.writeheader()
        writer.writerows(detailed_rows)

    # Save summary for ALL images
    all_summary = summarize_rows(detailed_rows, experiment_order)
    all_summary_path = PHASE6_DIR / "phase6_comparison_summary.csv"
    save_summary_csv(all_summary_path, all_summary)

    # Save summary for HELD-OUT TEST SET (20%)
    test_rows = [r for r in detailed_rows if r["split"] == "test"]
    test_summary = summarize_rows(test_rows, experiment_order)
    test_summary_path = PHASE6_DIR / "phase6_test_set_summary.csv"
    save_summary_csv(test_summary_path, test_summary)

    print_summary_table("PHASE 6 SUMMARY - ALL IMAGES", all_summary)
    print_summary_table("PHASE 6 SUMMARY - HELD-OUT TEST SET (20%)", test_summary)

    print(f"\nImages processed: {processed} | Skipped: {skipped}")
    print(f"All-data summary saved to:  {all_summary_path}")
    print(f"Test-set summary saved to:  {test_summary_path}")
    print("Phase 6 complete (all 6 proposal metrics + validation/test split).")


if __name__ == "__main__":
    main()