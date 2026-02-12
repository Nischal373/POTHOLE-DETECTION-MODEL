"""FastAPI service wrapper for pothole prediction."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
from pathlib import Path
import os
import shutil
import urllib.request
import re
from urllib.parse import urlparse, parse_qs

try:
    import gdown  # type: ignore
except Exception:  # pragma: no cover
    gdown = None

from predict import load_model, predict_image_with_model

MODEL_PATH = os.getenv("POTHOLE_MODEL_PATH", "best_pothole_model.pth")
MODEL_URL = os.getenv("POTHOLE_MODEL_URL")
DEFAULT_THRESHOLD = float(os.getenv("POTHOLE_THRESHOLD", "0.5"))

MODEL = None
CLASSES = None
DEVICE = None


def _extract_google_drive_file_id(url: str):
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if "id" in qs and qs["id"]:
        return qs["id"][0]

    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)

    return None


def _download_model(url: str, destination_path: str) -> None:
    dest = Path(destination_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    url_lower = url.lower()
    is_google_drive = "drive.google.com" in url_lower or "docs.google.com" in url_lower

    if is_google_drive:
        if gdown is None:
            raise RuntimeError(
                "POTHOLE_MODEL_URL looks like a Google Drive link, but 'gdown' is not installed. "
                "Add gdown to requirements and redeploy."
            )
        file_id = _extract_google_drive_file_id(url)
        try:
            if file_id:
                gdown.download(id=file_id, output=str(dest), quiet=False)
            else:
                # Fallback for uncommon Drive URL formats.
                gdown.download(url=url, output=str(dest), quiet=False, fuzzy=True)
        except Exception as exc:
            raise RuntimeError(
                "Failed to download model from Google Drive. Ensure the file is shared as 'Anyone with the link (Viewer)'. "
                "Use POTHOLE_MODEL_URL as https://drive.google.com/uc?export=download&id=<FILE_ID>. "
                "If Drive quota is exceeded, host the file on another source (Hugging Face/S3/R2)."
            ) from exc

        if not dest.exists() or dest.stat().st_size == 0:
            raise RuntimeError("Google Drive download did not produce a valid model file.")
        return

    # Download to a temp file first, then move into place.
    with tempfile.NamedTemporaryFile(delete=False, suffix=dest.suffix) as tmp:
        tmp_path = tmp.name
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pothole-detector-api"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp_path, "wb") as out:
            shutil.copyfileobj(resp, out)
        Path(tmp_path).replace(dest)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, CLASSES, DEVICE

    if not Path(MODEL_PATH).exists():
        if MODEL_URL:
            _download_model(MODEL_URL, MODEL_PATH)
        else:
            raise RuntimeError(
                f"Model file not found at '{MODEL_PATH}'. Either include it in the deployment, set POTHOLE_MODEL_PATH, or set POTHOLE_MODEL_URL to download it at startup."
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
