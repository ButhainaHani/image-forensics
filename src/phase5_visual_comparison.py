
import cv2
import csv
import re
import numpy as np
from pathlib import Path

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE = Path(__file__).resolve().parent.parent
OUTPUTS = BASE / "outputs"

DATASET = Path(r"C:\Datasets\CASIA2")

ORIGINAL_DIR = DATASET / "Tp"
GT_DIR = DATASET / "CASIA 2 Groundtruth"

ELA_DIR = OUTPUTS / "ela" / "ela_maps"
NOISE_DIR = OUTPUTS / "noise" / "noise_maps"
FREQUENCY_DIR = OUTPUTS / "phase5_experiments" / "frequency"
COMBINED_DIR = OUTPUTS / "phase5_experiments" / "ela_noise_frequency"

RESULT_DIR = OUTPUTS / "phase5_visual_comparisons"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. CHECK FOLDERS
# ==========================================

folders = {
    "Original": ORIGINAL_DIR,
    "Ground Truth": GT_DIR,
    "ELA": ELA_DIR,
    "Noise": NOISE_DIR,
    "Frequency": FREQUENCY_DIR,
    "Combined": COMBINED_DIR
}

print("\n===== FOLDER CHECK =====")

for name, folder in folders.items():
    print(f"{name}: {'FOUND' if folder.exists() else 'NOT FOUND'}")

if not ORIGINAL_DIR.exists() or not GT_DIR.exists():
    raise FileNotFoundError(
        "Original or Ground Truth folder was not found. Check dataset paths."
    )

if not COMBINED_DIR.exists():
    raise FileNotFoundError("Combined maps folder was not found.")

# ==========================================
# 3. FILE MATCHING
# ==========================================

def normalize_name(name):
    name = Path(name).stem.lower()

    for suffix in [
        "_combined", "_ela", "_noise", "_freq",
        "_fft", "_map", "_gt", "_mask",
        "_groundtruth", "_ground_truth"
    ]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]

    return name


def build_file_index(folder):
    index = {}

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in [
            ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
        ]:
            key = normalize_name(path.name)
            index.setdefault(key, path)

    return index


original_index = build_file_index(ORIGINAL_DIR)
gt_index = build_file_index(GT_DIR)
ela_index = build_file_index(ELA_DIR)
noise_index = build_file_index(NOISE_DIR)
frequency_index = build_file_index(FREQUENCY_DIR)

def find_matching_file(index, stem):
    key = normalize_name(stem)

    if key in index:
        return index[key]

    # Try matching names that contain the same complete base name.
    for indexed_name, path in index.items():
        if indexed_name.startswith(key) or key.startswith(indexed_name):
            if len(indexed_name) >= 12 and len(key) >= 12:
                return path

    return None


# ==========================================
# 4. IMAGE HELPERS
# ==========================================

def read_image(path, grayscale=False):
    if path is None:
        return None

    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    return cv2.imread(str(path), flag)


def resize_image(image, size):
    if image is None:
        return None

    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def make_placeholder(size, message):
    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)

    cv2.putText(
        image, message, (10, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (255, 255, 255), 1, cv2.LINE_AA
    )

    return image


def colorize_map(image, size):
    if image is None:
        return make_placeholder(size, "MAP NOT FOUND")

    image = resize_image(image, size)
    return cv2.applyColorMap(image, cv2.COLORMAP_JET)


def add_title(image, title):
    height, width = image.shape[:2]

    canvas = np.zeros((height + 40, width, 3), dtype=np.uint8)
    canvas[40:, :] = image

    cv2.putText(
        canvas, title, (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (255, 255, 255), 2, cv2.LINE_AA
    )

    return canvas


# ==========================================
# 5. PROCESS ALL COMBINED MAPS
# ==========================================

combined_files = sorted(COMBINED_DIR.glob("*.png"))

if not combined_files:
    raise RuntimeError("No combined PNG maps found.")

success = 0
failed = []
missing_maps = []

print("\n===== STARTING VISUAL COMPARISON =====")
print("Combined maps found:", len(combined_files))

for count, combined_path in enumerate(combined_files, 1):

    stem = normalize_name(combined_path.name)

    original_path = find_matching_file(original_index, stem)
    gt_path = find_matching_file(gt_index, stem)

    ela_path = find_matching_file(ela_index, stem)
    noise_path = find_matching_file(noise_index, stem)
    frequency_path = find_matching_file(frequency_index, stem)

    original = read_image(original_path)
    ground_truth = read_image(gt_path, True)

    combined = read_image(combined_path, True)
    ela = read_image(ela_path, True)
    noise = read_image(noise_path, True)
    frequency = read_image(frequency_path, True)

    if original is None or ground_truth is None:
        reason = []

        if original is None:
            reason.append("Original missing")

        if ground_truth is None:
            reason.append("Ground Truth missing")

        failed.append([combined_path.name, "; ".join(reason)])
        continue

    height, width = original.shape[:2]
    size = (width, height)

    ground_truth = resize_image(ground_truth, size)

    panels = [
        add_title(original, "Original Image"),
        add_title(colorize_map(ground_truth, size), "Ground Truth"),
        add_title(colorize_map(ela, size), "ELA"),
        add_title(colorize_map(noise, size), "Noise"),
        add_title(colorize_map(frequency, size), "Frequency"),
        add_title(colorize_map(combined, size), "Combined Forensic Map")
    ]

    if ela is None or noise is None or frequency is None:
        missing_maps.append(combined_path.name)

    row1 = cv2.hconcat(panels[:3])
    row2 = cv2.hconcat(panels[3:])

    comparison = cv2.vconcat([row1, row2])

    output_path = RESULT_DIR / f"{stem}_comparison.png"

    if cv2.imwrite(str(output_path), comparison):
        success += 1
    else:
        failed.append([combined_path.name, "Could not save comparison"])

    if count % 250 == 0:
        print(f"Processed {count}/{len(combined_files)}")


# ==========================================
# 6. SAVE REPORT
# ==========================================

with open(
    RESULT_DIR / "comparison_report.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)
    writer.writerow(["Image", "Problem"])
    writer.writerows(failed)

with open(
    RESULT_DIR / "missing_maps_report.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)
    writer.writerow(["Image"])
    for name in missing_maps:
        writer.writerow([name])


# ==========================================
# 7. FINAL SUMMARY
# ==========================================

print("\n========== VISUAL COMPARISON SUMMARY ==========")
print("Total combined maps:", len(combined_files))
print("Comparisons saved:", success)
print("Images failed:", len(failed))
print("Images with missing individual maps:", len(missing_maps))
print("Output folder:", RESULT_DIR)
print("Process finished.")
