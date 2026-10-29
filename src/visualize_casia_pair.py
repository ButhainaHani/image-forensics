
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

root = Path(r"C:\Datasets\CASIA2")
image_name = "Tp_D_CNN_M_N_sec00011_cha00085_11227.jpg"

image_path = root / "Tp" / image_name
mask_path = root / "CASIA 2 Groundtruth" / f"{Path(image_name).stem}_gt.png"

image = cv2.imread(str(image_path))
mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")
if mask is None:
    raise FileNotFoundError(f"Mask not found: {mask_path}")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Rotate the mask for visualization only
rotated_mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)

print("Image shape:", image.shape[:2])
print("Original mask shape:", mask.shape)
print("Rotated mask shape:", rotated_mask.shape)

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

axes[0].imshow(image)
axes[0].set_title("Tampered Image")

axes[1].imshow(rotated_mask, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Rotated Mask (Preview Only)")

axes[2].imshow(image)
if image.shape[:2] == rotated_mask.shape:
    axes[2].imshow(rotated_mask > 127, cmap="Reds", alpha=0.45)
    axes[2].set_title("Rotated Mask Overlay")
else:
    axes[2].set_title("Dimensions Still Differ")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

output = Path(__file__).resolve().parent.parent / "outputs"
output.mkdir(parents=True, exist_ok=True)

save_path = output / "casia_rotation_test.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print("Saved preview:", save_path)

plt.show()
