"""FastAPI service wrapper for pothole prediction."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
from pathlib import Path
import os

from predict import load_model, predict_image_with_model

MODEL_PATH = os.getenv("POTHOLE_MODEL_PATH", "best_pothole_model.pth")
DEFAULT_THRESHOLD = float(os.getenv("POTHOLE_THRESHOLD", "0.5"))

MODEL = None
CLASSES = None
DEVICE = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, CLASSES, DEVICE
    if not Path(MODEL_PATH).exists():
        raise RuntimeError(
            f"Model file not found at '{MODEL_PATH}'. Set POTHOLE_MODEL_PATH or include the file in the deployment."
        )
    MODEL, CLASSES, DEVICE = load_model(MODEL_PATH)
    yield


app = FastAPI(title="Pothole Detector API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if MODEL is None or CLASSES is None or DEVICE is None:
        return JSONResponse(
            {"success": False, "error": "Model not initialized"}, status_code=500
        )

    if not file.filename:
        return JSONResponse(
            {"success": False, "error": "No file provided"}, status_code=400
        )

    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(file.file.read())
        temp_path = temp.name

    try:
        result = predict_image_with_model(
            temp_path,
            model=MODEL,
            classes=CLASSES,
            device=DEVICE,
            threshold=DEFAULT_THRESHOLD,
        )
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    status_code = 200 if result.get("success") else 400
    return JSONResponse(result, status_code=status_code)
