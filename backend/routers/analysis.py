import uuid
from fastapi import APIRouter, File, HTTPException, UploadFile
from backend.config import UPLOAD_DIR
from backend.schemas import CombinedAnalysisResponse, SingleAnalysisResponse
from backend.services.forensics_service import (
    run_combined_analysis,
    run_ela_analysis,
    run_frequency_analysis,
    run_noise_analysis,
)
from backend.services.validation import validate_image_file

router = APIRouter()


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    content = await file.read()

    try:
        ext, (width, height) = validate_image_file(file.filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    file_id = f"{uuid.uuid4().hex}{ext}"
    saved_path = UPLOAD_DIR / file_id
    saved_path.write_bytes(content)

    return {
        "file_id": file_id,
        "original_filename": file.filename,
        "width": width,
        "height": height,
        "size_bytes": len(content),
        "saved_path": str(saved_path.relative_to(UPLOAD_DIR.parent.parent)),
    }


@router.post("/analyze/ela/{file_id}", response_model=SingleAnalysisResponse)
def analyze_ela(file_id: str):
    try:
        return run_ela_analysis(file_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AttributeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ELA analysis failed: {exc}") from exc


@router.post("/analyze/noise/{file_id}", response_model=SingleAnalysisResponse)
def analyze_noise(file_id: str):
    try:
        return run_noise_analysis(file_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AttributeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Noise analysis failed: {exc}") from exc


@router.post("/analyze/frequency/{file_id}", response_model=SingleAnalysisResponse)
def analyze_frequency(file_id: str):
    try:
        return run_frequency_analysis(file_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AttributeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Frequency analysis failed: {exc}") from exc


@router.post("/analyze/combined/{file_id}", response_model=CombinedAnalysisResponse)
def analyze_combined(file_id: str):
    try:
        return run_combined_analysis(file_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Combined analysis failed: {exc}") from exc