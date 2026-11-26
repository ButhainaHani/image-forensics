from io import BytesIO
from pathlib import Path
from PIL import Image, UnidentifiedImageError

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB


def validate_image_file(filename: str | None, content: bytes) -> tuple[str, tuple[int, int]]:
    if not content:
        raise ValueError("Uploaded file is empty.")

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise ValueError("File size exceeds the 15 MB limit.")

    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )

    try:
        with Image.open(BytesIO(content)) as img:
            img.verify()
        with Image.open(BytesIO(content)) as img:
            width, height = img.size
    except (UnidentifiedImageError, Exception) as exc:
        raise ValueError("Invalid or corrupted image file.") from exc

    if width < 16 or height < 16:
        raise ValueError("Image dimensions are too small for forensic analysis.")

    return ext, (width, height)