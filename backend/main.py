from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import analysis
from fastapi.staticfiles import StaticFiles
from backend.config import RESULTS_DIR

app = FastAPI(
    title="Digital Image Forensics API",
    version="1.0.0",
)

outputs_root = RESULTS_DIR.parent
outputs_root.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(outputs_root)), name="outputs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis.router, prefix="/api/v1", tags=["Analysis"])


@app.get("/health")
def health_check():
    return {"status": "ok", "phase": 7}