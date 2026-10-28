
from pathlib import Path

root = Path(r"C:\Datasets\CASIA2")
tp_folder = root / "Tp"
gt_folder = root / "CASIA 2 Groundtruth"

tp_files = {
    p.stem: p
    for p in tp_folder.iterdir()
    if p.is_file()
}

gt_files = {
    p.stem.removesuffix("_gt"): p
    for p in gt_folder.iterdir()
    if p.is_file() and p.stem.endswith("_gt")
}

matched = []
unmatched = []

for name, path in tp_files.items():
    if name in gt_files:
        matched.append((path.name, gt_files[name].name))
    else:
        unmatched.append(path.name)

print("Total tampered images:", len(tp_files))
print("Ground-truth masks:", len(gt_files))
print("Matched pairs:", len(matched))
print("Images without exact matching mask:", len(unmatched))

print("\nMatched examples:")
for image_name, mask_name in matched[:15]:
    print(f"Image: {image_name}  |  Mask: {mask_name}")

print("\nUnmatched examples:")
for name in unmatched[:15]:
    print(name)
