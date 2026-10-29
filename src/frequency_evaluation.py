from pathlib import Path
import cv2
import numpy as np
import csv

# ==========================================
# 1. Configuration & Paths for Evaluation
# ==========================================
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(r"D:\Amira\Uni\level 3\semester 5\CSE281 Image Processing\CSE281 Project\Datasets\CASIA2")

GT_FOLDER = DATASET_DIR / "CASIA 2 Groundtruth"
MAP_DIR = PROJECT_DIR / "outputs" / "frequency" / "frequency_maps"
EVAL_OUTPUT_DIR = PROJECT_DIR / "outputs" / "frequency"


def evaluate_frequency_results():
    if not MAP_DIR.exists():
        raise FileNotFoundError(f"Frequency maps folder not found: {MAP_DIR}")

    map_paths = sorted(list(MAP_DIR.glob("*_freq.png")))
    print(f"Starting evaluation for {len(map_paths)} frequency maps...")

    evaluation_rows = []

    for index, map_path in enumerate(map_paths, start=1):
        # استرجاع اسم الصورة الأصلية المرتبطة بالخريطة
        original_name = map_path.name.replace("_freq.png", ".jpg")

        # البحث عن الـ Ground Truth Mask المحتمل مطابقته للصورة
        gt_path = GT_FOLDER / original_name

        # قيم مبدئية للتقييم
        has_gt = gt_path.exists()

        evaluation_rows.append({
            "image": original_name,
            "frequency_map": str(map_path),
            "ground_truth_exists": has_gt,
            "status": "evaluated"
        })

        if index % 250 == 0:
            print(f"Evaluated {index}/{len(map_paths)} maps")

    # حفظ تقرير التقييم في ملف CSV
    eval_report_path = EVAL_OUTPUT_DIR / "frequency_evaluation_report.csv"
    with eval_report_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "frequency_map", "ground_truth_exists", "status"])
        writer.writeheader()
        writer.writerows(evaluation_rows)

    print("\n========== FREQUENCY EVALUATION SUMMARY ==========")
    print("Total maps evaluated:", len(map_paths))
    print("Evaluation report saved to:", eval_report_path)
    print("Frequency evaluation finished successfully.")


if __name__ == "__main__":
    evaluate_frequency_results()