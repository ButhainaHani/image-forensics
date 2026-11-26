from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
UPLOAD_DIR = BASE_DIR / "outputs" / "api_uploads"
RESULTS_DIR = BASE_DIR / "outputs" / "api_results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)