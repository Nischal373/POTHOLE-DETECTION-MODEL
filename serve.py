"""
FastAPI service wrapper for pothole prediction.
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
from pathlib import Path
import os

from predict import predict_image

app = FastAPI(title="Pothole Detector API")

MODEL_PATH = os.getenv("POTHOLE_MODEL_PATH", "best_pothole_model.pth")
DEFAULT_THRESHOLD = float(os.getenv("POTHOLE_THRESHOLD", "0.5"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if not file.filename:
        return JSONResponse(
            {"success": False, "error": "No file provided"}, status_code=400
        )

    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(file.file.read())
        temp_path = temp.name

    try:
        result = predict_image(temp_path, MODEL_PATH, threshold=DEFAULT_THRESHOLD)
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    status_code = 200 if result.get("success") else 400
    return JSONResponse(result, status_code=status_code)
