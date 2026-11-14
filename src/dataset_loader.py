from pathlib import Path
import cv2


def load_image(image_path):
    """Load an image and return it in RGB format."""
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mask(mask_path):
    """Load a manipulation mask as a grayscale image."""
    mask_path = Path(mask_path)

    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")

    return mask


if __name__ == "__main__":
    eraser_path = Path(r"C:\Datasets\eraser_benchmark_folders")

    # Test one image and its mask
    image_path = eraser_path / "15" / "15_.png"
    mask_path = eraser_path / "15" / "15_01.png"

    image = load_image(image_path)
    mask = load_mask(mask_path)


    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("Image loaded successfully!")
    print("Image shape:", image.shape)
    print("Mask loaded successfully!")
    print("Mask shape:", mask.shape)

