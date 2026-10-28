
import cv2
import numpy as np


def preprocess_image(image):
    """Validate an image and prepare it for analysis."""
    if image is None:
        raise ValueError("Input image is empty.")

    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a 3-channel color image.")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image.copy()


def preprocess_mask(mask):
    """Convert a ground-truth mask to a binary mask."""
    if mask is None:
        raise ValueError("Input mask is empty.")

    if not isinstance(mask, np.ndarray):
        raise TypeError("Input mask must be a NumPy array.")

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if mask.ndim != 2:
        raise ValueError("Expected a grayscale mask.")

    return (mask > 127).astype(np.uint8)


def validate_image_mask(image, mask):
    """Ensure image and ground-truth mask are aligned."""
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Dimension mismatch: image={image.shape[:2]}, "
            f"mask={mask.shape[:2]}"
        )

    return True


if __name__ == "__main__":
    from pathlib import Path

    dataset_path = Path(r"C:\Datasets\CASIA2")
    image_dir = dataset_path / "Tp"
    mask_dir = dataset_path / "CASIA 2 Groundtruth"

    # Find the first available tampered image
    image_files = [
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
        }
    ]

    if not image_files:
        raise FileNotFoundError("No images found inside Tp.")

    image_path = image_files[0]

    # Load the image
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    processed_image = preprocess_image(image)

    print("Image preprocessing successful!")
    print("Image:", image_path.name)
    print("Image shape:", processed_image.shape)

    print("\nNow we need to identify its matching ground-truth mask.")
    print("Mask folder exists:", mask_dir.exists())
