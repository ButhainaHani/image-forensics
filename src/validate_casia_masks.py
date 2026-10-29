
from pathlib import Path
import cv2
import numpy as np

root = Path(r"C:\Datasets\CASIA2")
tp_folder = root / "Tp"
gt_folder = root / "CASIA 2 Groundtruth"

masks = {
    p.stem.removesuffix("_gt"): p
    for p in gt_folder.iterdir()
    if p.is_file() and p.stem.endswith("_gt")
}

checked = 0

for image_path in tp_folder.iterdir():
    if not image_path.is_file():
        continue

    if image_path.stem not in masks:
        continue

    image = cv2.imread(str(image_path))
    mask_path = masks[image_path.stem]
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    print(f"\nImage: {image_path.name}")
    print(f"Mask:  {mask_path.name}")

    if image is None or mask is None:
        print("ERROR: Could not read image or mask.")
        continue

    print("Image dimensions (height, width):", image.shape[:2])
    print("Mask dimensions (height, width):", mask.shape)
    print("Dimensions match:", image.shape[:2] == mask.shape)
    print("Unique mask values:", np.unique(mask)[:20])

    if image.shape[:2] != mask.shape:
        print("WARNING: Dimensions do not match.")

    checked += 1
    if checked == 5:
        break

print(f"\nFinished checking {checked} matched pairs.")
