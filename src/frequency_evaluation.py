
from pathlib import Path
import cv2
import numpy as np
import csv

# ==========================================
# 1. Configuration & Paths for Evaluation
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Your PC Dataset Path
DATASET_DIR = Path(r"C:\Datasets\CASIA2")

GT_FOLDER = DATASET_DIR / "CASIA 2 Groundtruth"

MAP_DIR = PROJECT_DIR / "outputs" / "frequency" / "frequency_maps"

EVAL_OUTPUT_DIR = PROJECT_DIR / "outputs" / "frequency"


def evaluate_frequency_results():

    if not MAP_DIR.exists():
        raise FileNotFoundError(
            f"Frequency maps folder not found: {MAP_DIR}"
        )

    if not GT_FOLDER.exists():
        raise FileNotFoundError(
            f"Ground Truth folder not found: {GT_FOLDER}"
        )

    map_paths = sorted(list(MAP_DIR.glob("*_freq.png")))

    print(f"Starting evaluation for {len(map_paths)} frequency maps...")

    evaluation_rows = []

    for index, map_path in enumerate(map_paths, start=1):

        # Retrieve corresponding Ground Truth mask name
        original_name = map_path.name.replace("_freq.png", "_gt.png")

        # Find corresponding Ground Truth Mask
        gt_path = GT_FOLDER / original_name

        # Check if Ground Truth exists
        has_gt = gt_path.exists()

        evaluation_rows.append({
            "image": original_name.replace("_gt.png", ".jpg"),
            "frequency_map": str(map_path),
            "ground_truth_exists": has_gt,
            "status": "evaluated"
        })

        if index % 250 == 0:
            print(f"Evaluated {index}/{len(map_paths)} maps")

    # ==========================================
    # 2. Ground Truth Matching Summary
    # ==========================================

    matched_count = sum(
        row["ground_truth_exists"] for row in evaluation_rows
    )

    missing_count = len(evaluation_rows) - matched_count

    print("\n========== GROUND TRUTH MATCHING ==========")
    print("Matched Ground Truth:", matched_count)
    print("Missing Ground Truth:", missing_count)

    # ==========================================
    # 3. Save Evaluation Report
    # ==========================================

    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    eval_report_path = EVAL_OUTPUT_DIR / "frequency_evaluation_report.csv"

    with eval_report_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "frequency_map",
                "ground_truth_exists",
                "status"
            ]
        )

        writer.writeheader()
        writer.writerows(evaluation_rows)

    # ==========================================
    # 4. Final Summary
    # ==========================================

    print("\n========== FREQUENCY EVALUATION SUMMARY ==========")
    print("Total maps evaluated:", len(map_paths))
    print("Matched Ground Truth:", matched_count)
    print("Missing Ground Truth:", missing_count)
    print("Evaluation report saved to:", eval_report_path)
    print("Frequency evaluation finished successfully.")


if __name__ == "__main__":
    evaluate_frequency_results()
