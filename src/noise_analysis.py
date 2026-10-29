from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from dataset_loader import load_image, load_mask


def extract_noise_residual(image_rgb: np.ndarray, ksize: int = 3) -> np.ndarray:
    """استخراج متبقي النويز عالي التردد"""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    smoothed = cv2.medianBlur(gray.astype(np.uint8), ksize).astype(np.float32)
    residual = gray - smoothed
    return residual


def compute_local_noise_map(residual: np.ndarray, window_size: int = 7) -> np.ndarray:
    """حساب خريطة التباين المحلي للنويز مع تعزيز التباين البصري"""
    mean = uniform_filter(residual, size=window_size)
    mean_sq = uniform_filter(residual ** 2, size=window_size)
    variance = np.maximum(mean_sq - (mean ** 2), 0)

    log_variance = np.log1p(variance)
    norm_map = cv2.normalize(log_variance, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_map = clahe.apply(norm_map.astype(np.uint8))

    return enhanced_map


def generate_noise_heatmap(noise_map: np.ndarray) -> np.ndarray:
    """تلوين خريطة النويز لتوضيح الأماكن المشبوهة كألوان حرارية"""
    return cv2.applyColorMap(noise_map, cv2.COLORMAP_JET)


def process_and_save_dataset(dataset_dir: Path, output_dir: Path, limit: int = None):
    """
    معالجة مجموعة الصور على دفعات (Batch Processing) وحفظ خرائط النويز والخرائط الحرارية
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = output_dir / "heatmaps"
    gray_maps_dir = output_dir / "noise_maps"
    heatmap_dir.mkdir(exist_ok=True)
    gray_maps_dir.mkdir(exist_ok=True)

    # البحث عن كل الصور بصيغة png التي تمثل الصورة المعدلة (15_.png)
    image_paths = sorted(list(dataset_dir.glob("*/*_.png")))
    if not image_paths:
        # محاولة بديلة في حال كانت الأسماء بدون underscore
        image_paths = sorted(list(dataset_dir.glob("*/*.png")))

    if limit:
        image_paths = image_paths[:limit]

    print(f"Found {len(image_paths)} images to process...")

    for idx, img_path in enumerate(image_paths, 1):
        try:
            folder_name = img_path.parent.name
            img = load_image(img_path)

            residual = extract_noise_residual(img)
            noise_map = compute_local_noise_map(residual)
            heatmap = generate_noise_heatmap(noise_map)

            # حفظ خريطة النويز الرمادية والخريطة الحرارية
            cv2.imwrite(str(gray_maps_dir / f"{folder_name}_noise.png"), noise_map)
            cv2.imwrite(str(heatmap_dir / f"{folder_name}_heatmap.png"), heatmap)

            print(f"[{idx}/{len(image_paths)}] Processed: Folder {folder_name}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"\nAll results saved successfully to: {output_dir}")


def main():
    base_dataset = Path(
        r"C:\Users\AdminOS\OneDrive\Documents\datasets\eraser_benchmark_folders\eraser_benchmark_folders")

    # مجلد حفظ المخرجات داخل المشروع
    results_dir = Path(__file__).resolve().parent.parent / "results" / "noise"

    # تشغيل المعالجة لأول 5 مجلدات للتجربة السريعة (يمكنك إزالة limit=5 لمعالجة الداتاسيت كاملة)
    process_and_save_dataset(base_dataset, results_dir, limit=5)


if __name__ == "__main__":
    main()