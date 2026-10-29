from pathlib import Path
import cv2
import numpy as np
from dataset_loader import load_image, load_mask
from noise_analysis import extract_noise_residual, compute_local_noise_map


def evaluate_noise_detection(noise_map: np.ndarray, gt_mask: np.ndarray, threshold: int = 128):
    """
    تحويل خريطة النويز لقناع ثنائي ومقارنته بالـ Ground Truth لحساب مقاييس الدقة.
    """
    _, binary_pred = cv2.threshold(noise_map, threshold, 255, cv2.THRESH_BINARY)
    pred_bin = (binary_pred > 0).astype(np.uint8)

    gt_bin = (gt_mask > 0).astype(np.uint8)

    if pred_bin.shape != gt_bin.shape:
        pred_bin = cv2.resize(pred_bin, (gt_bin.shape[1], gt_bin.shape[0]), interpolation=cv2.INTER_NEAREST)

    tp = np.sum((pred_bin == 1) & (gt_bin == 1))
    fp = np.sum((pred_bin == 1) & (gt_bin == 0))
    fn = np.sum((pred_bin == 0) & (gt_bin == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    intersection = tp
    union = np.sum((pred_bin == 1) | (gt_bin == 1))
    iou = intersection / union if union > 0 else 0.0

    return {
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1, 4),
        "IoU": round(iou, 4)
    }


def main():
    base_path = Path(r"C:\Users\AdminOS\OneDrive\Documents\datasets\eraser_benchmark_folders\eraser_benchmark_folders\15")
    image_path = base_path / "15_.png"
    mask_path = base_path / "15_01.png"

    image = load_image(image_path)
    gt_mask = load_mask(mask_path)

    for k in [3, 5]:
        residual = extract_noise_residual(image, ksize=k)
        noise_map = compute_local_noise_map(residual)
        metrics = evaluate_noise_detection(noise_map, gt_mask)

        print(f"--- Results for Filter Size k={k} ---")
        for metric_name, val in metrics.items():
            print(f"{metric_name}: {val * 100:.2f}%")
        print()


if __name__ == "__main__":
    main()