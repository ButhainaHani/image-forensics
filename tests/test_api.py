import io
from fastapi.testclient import TestClient
from PIL import Image
from backend.main import app

client = TestClient(app)


def test_health_and_upload_and_missing_file_404():
    health_res = client.get("/health")
    assert health_res.status_code == 200

    img = Image.new("RGB", (64, 64), color=(120, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    res = client.post(
        "/api/v1/upload",
        files={"file": ("sample.png", buf, "image/png")},
    )
    assert res.status_code == 200
    data = res.json()
    assert "file_id" in data

    missing_res = client.post("/api/v1/analyze/ela/nonexistent_file.png")
    assert missing_res.status_code == 404


def test_invalid_file_rejected_400():
    bad_file = io.BytesIO(b"not an image content")
    res = client.post(
        "/api/v1/upload",
        files={"file": ("notes.txt", bad_file, "text/plain")},
    )
    assert res.status_code == 400