from typing import Any
from pydantic import BaseModel, Field


class SingleAnalysisResponse(BaseModel):
    module: str = Field(..., examples=["ela", "noise", "frequency"])
    file_id: str
    job_id: str
    output_image: str | None = None
    details: dict[str, Any] | None = None


class CombinedAnalysisResponse(BaseModel):
    file_id: str
    job_id: str
    ela: SingleAnalysisResponse
    noise: SingleAnalysisResponse
    frequency: SingleAnalysisResponse
    combined_heatmap: str | None = None
    details: dict[str, Any] | None = None