
from pathlib import Path
import cv2
import numpy as np
import csv


# =========================
# PATHS
# =========================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = Path(r"C:\Datasets\CASIA2")
GT_FOLDER = DATASET_DIR / "CASIA 2 Groundtruth"

NOISE_MAP_DIR = PROJECT_DIR / "outputs" / "noise" / "noise_maps"

REPORT_PATH = (
    PROJECT_DIR / "outputs" / "noise" /
    "noise_evaluation_report.csv"
)

THRESHOLD = 128


# =========================
# EVALUATION
# =========================

def evaluate_noise_detection(noise_map, gt_mask):

    _, binary_pred = cv2.threshold(
        noise_map,
        THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    pred = binary_pred > 0
    gt = gt_mask > 0

    if pred.shape != gt.shape:
        pred = cv2.resize(
            pred.astype(np.uint8),
            (gt.shape[1], gt.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ) > 0

    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    union = np.sum(pred | gt)
    iou = tp / union if union else 0.0

    return {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "IoU": iou
    }


# =========================
# MAIN
# =========================

def main():

    map_paths = sorted(NOISE_MAP_DIR.glob("*_noise.png"))

    print(f"Found {len(map_paths)} noise maps.")

    rows = []

    matched_count = 0
    missing_count = 0

    for idx, map_path in enumerate(map_paths, 1):

        original_stem = map_path.stem.replace("_noise", "")

        gt_path = GT_FOLDER / f"{original_stem}_gt.png"

        noise_map = cv2.imread(
            str(map_path),
            cv2.IMREAD_GRAYSCALE
        )

        if noise_map is None:
            rows.append({
                "image": original_stem,
                "status": "noise map unreadable"
            })
            continue

        if not gt_path.exists():

            missing_count += 1

            rows.append({
                "image": original_stem,
                "status": "ground truth missing"
            })

            continue

        gt_mask = cv2.imread(
            str(gt_path),
            cv2.IMREAD_GRAYSCALE
        )

        if gt_mask is None:
            rows.append({
                "image": original_stem,
                "status": "ground truth unreadable"
            })
            continue

        metrics = evaluate_noise_detection(noise_map, gt_mask)

        matched_count += 1

        rows.append({
            "image": original_stem,
            **metrics,
            "status": "evaluated"
        })

        if idx % 250 == 0:
            print(f"Evaluated {idx}/{len(map_paths)} maps")

    # Save evaluation report

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image",
        "Precision",
        "Recall",
        "F1-Score",
        "IoU",
        "status"
    ]

    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(rows)

    # Calculate average metrics

    evaluated_rows = [
        row for row in rows
        if row.get("status") == "evaluated"
    ]

    print("\n========== NOISE EVALUATION SUMMARY ==========")
    print(f"Total noise maps: {len(map_paths)}")
    print(f"Matched Ground Truth: {matched_count}")
    print(f"Missing Ground Truth: {missing_count}")

    if evaluated_rows:

        for metric in [
            "Precision",
            "Recall",
            "F1-Score",
            "IoU"
        ]:

            average = np.mean([
                row[metric] for row in evaluated_rows
            ])

            print(f"Average {metric}: {average:.4f}")

    print(f"Evaluation report: {REPORT_PATH}")
    print("Noise evaluation finished.")


if __name__ == "__main__":
    main()
