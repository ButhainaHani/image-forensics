from pathlib import Path
import cv2
import numpy as np
import csv

# ==========================================
# 1. Project Configuration & Paths
# ==========================================
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(r"D:\Amira\Uni\level 3\semester 5\CSE281 Image Processing\CSE281 Project\Datasets\CASIA2")

TP_FOLDER = DATASET_DIR / "Tp"
GT_FOLDER = DATASET_DIR / "CASIA 2 Groundtruth"

# مسار المخرجات الخاصة بالـ Frequency Analysis
OUTPUT_DIR = PROJECT_DIR / "outputs" / "frequency"
MAP_DIR = OUTPUT_DIR / "frequency_maps"

# إنشاء المجلدات لو مش موجودة
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAP_DIR.mkdir(parents=True, exist_ok=True)

print("Frequency Analysis environment is ready!", flush=True)


# ==========================================
# 2. Frequency Analysis Core (DCT-based)
# ==========================================
def perform_frequency_analysis(image_path):
    """
    تحويل الصورة إلى التدرج الرمادي وتطبيق Discrete Cosine Transform (DCT)
    لاستخراج خريطة الترددات وكشف آثار التلاعب أو إعادة الضغط.
    """
    # قراءة الصورة
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # تحويل الصورة إلى Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # تحويل القيم إلى float32 لعملية الـ DCT
    gray_float = np.float32(gray) / 255.0

    # تطبيق الـ 2D DCT
    dct_image = cv2.dct(gray_float)

    # التركيز على الترددات العالية أو استخراج خريطة طاقة الترددات (Energy Map)
    # لحساب خريطة مرئية، يمكننا حساب القيمة المطلقة وتضخيمها للعرض
    dct_log = np.log(np.abs(dct_image) + 1.0)

    # تطبيع القيم لتكون بين 0 و 255 من نوع uint8 لحفظها كصورة
    dct_normalized = cv2.normalize(dct_log, None, 0, 255, cv2.NORM_MINMAX)
    frequency_map = dct_normalized.astype(np.uint8)

    return frequency_map


# ==========================================
# 3. Main Processing Function
# ==========================================
def main():
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    if not TP_FOLDER.exists():
        raise FileNotFoundError(f"Image folder not found: {TP_FOLDER}")

    image_paths = sorted(
        p for p in TP_FOLDER.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(f"Starting Frequency Analysis on {len(image_paths)} images...")

    report_rows = []
    success_count = 0

    for index, image_path in enumerate(image_paths, start=1):
        try:
            # تطبيق دالة التحليل الترددي
            freq_map = perform_frequency_analysis(image_path)

            # مسار حفظ الخريطة الترددية
            output_name = image_path.stem + "_freq.png"
            freq_map_path = MAP_DIR / output_name

            # حفظ خريطة الترددات كصورة
            cv2.imwrite(str(freq_map_path), freq_map)

            report_rows.append({
                "image": image_path.name,
                "frequency_map": str(freq_map_path),
                "status": "success"
            })
            success_count += 1

        except Exception as error:
            report_rows.append({
                "image": image_path.name,
                "frequency_map": "",
                "status": f"error: {error}"
            })

        if index % 250 == 0:
            print(f"Processed {index}/{len(image_paths)} images")

    # حفظ تقرير الـ CSV
    report_path = OUTPUT_DIR / "frequency_report.csv"
    with report_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "frequency_map", "status"])
        writer.writeheader()
        writer.writerows(report_rows)

    print("\n========== FREQUENCY ANALYSIS SUMMARY ==========")
    print("Total images processed:", len(image_paths))
    print("Successful frequency maps:", success_count)
    print("Output folder:", OUTPUT_DIR)
    print("Report saved to:", report_path)
    print("Frequency analysis finished successfully.")


if __name__ == "__main__":
    main()