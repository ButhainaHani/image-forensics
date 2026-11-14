
from pathlib import Path
import csv
import cv2

ROOT = Path(r"C:\Datasets\CASIA2")
TP_FOLDER = ROOT / "Tp"
GT_FOLDER = ROOT / "CASIA 2 Groundtruth"

OUTPUT = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"
}

print("Starting CASIA2 audit...", flush=True)

# Index ground-truth masks by the image basename.
masks = {
    p.stem[:-3]: p
    for p in GT_FOLDER.iterdir()
    if p.is_file()
    and p.stem.lower().endswith("_gt")
    and p.suffix.lower() in IMAGE_EXTENSIONS
}

image_files = sorted(
    p for p in TP_FOLDER.iterdir()
    if p.is_file()
    and p.suffix.lower() in IMAGE_EXTENSIONS
)

report = []
matched = 0
valid = 0
unmatched = 0

for index, image_path in enumerate(image_files, start=1):
    mask_path = masks.get(image_path.stem)

    row = {
        "image": image_path.name,
        "mask": mask_path.name if mask_path else "",
        "status": "",
        "image_shape": "",
        "mask_shape": "",
    }

    if mask_path is None:
        row["status"] = "NO_MATCHING_MASK"
        unmatched += 1
        report.append(row)
        continue

    matched += 1
    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        row["status"] = "IMAGE_READ_ERROR"
    elif mask is None:
        row["status"] = "MASK_READ_ERROR"
        row["image_shape"] = str(image.shape[:2])
    else:
        row["image_shape"] = str(image.shape[:2])
        row["mask_shape"] = str(mask.shape)

        if image.shape[:2] != mask.shape:
            row["status"] = "DIMENSION_MISMATCH"
        else:
            row["status"] = "OK"
            valid += 1

    report.append(row)

    if index % 500 == 0:
        print(f"Checked {index}/{len(image_files)} images")

report_path = OUTPUT / "casia_audit_report.csv"

with report_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image", "mask", "status",
            "image_shape", "mask_shape"
        ],
    )
    writer.writeheader()
    writer.writerows(report)

other_issues = sum(
    row["status"] not in ("OK", "NO_MATCHING_MASK")
    for row in report
)

print("\nCASIA2 AUDIT FINISHED")
print("Image files checked:", len(image_files))
print("Matching image-mask names:", matched)
print("Readable pairs with matching dimensions:", valid)
print("Images without matching masks:", unmatched)
print("Other issues:", other_issues)
print("Report saved to:", report_path)
