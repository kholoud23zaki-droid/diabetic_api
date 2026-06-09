import asyncio
import io
import os
import threading

import numpy as np
import tensorflow as tf
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────────────
IMAGE_SIZE       = (224, 224)
CLASS_NAMES      = ["Non_Ulcer", "Ulcer"]
LOCAL_MODEL_PATH = "/app/diabetic_foot_uIcer.keras"   # /app is writable on Railway
MAX_UPLOAD_MB    = 10
GDRIVE_FILE_ID   = os.getenv("GDRIVE_FILE_ID", "1rH0ix5iuYgTLyCramttS4gOUSC1TvBku")

LANCZOS = getattr(Image, "Resampling", Image).LANCZOS

model      = None
model_lock = threading.Lock()


# ── Download ───────────────────────────────────────────────────────────────────
def download_from_gdrive(file_id: str, dest: str) -> None:
    import gdown

    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print(f"Model already cached at {dest} ({os.path.getsize(dest)/1e6:.1f} MB), skipping download.")
        return

    print(f"Downloading model from Google Drive (id={file_id}) to {dest} ...")
    url = f"https://drive.google.com/uc?id={file_id}"

    # Download directly to final destination — no .tmp rename needed
    try:
        result = gdown.download(url, dest, quiet=False, fuzzy=True)
    except Exception as exc:
        if os.path.exists(dest):
            os.remove(dest)
        raise RuntimeError(f"gdown raised an exception: {exc}") from exc

    if result is None or not os.path.exists(dest):
        raise RuntimeError(
            "gdown returned None — download failed. "
            "Check that the file is shared as 'Anyone with the link'."
        )

    size_mb = os.path.getsize(dest) / 1e6
    if size_mb < 1:
        os.remove(dest)
        raise RuntimeError(
            f"Downloaded file is only {size_mb:.2f} MB — looks like an HTML error page."
        )

    print(f"Model downloaded to {dest} ({size_mb:.1f} MB)")


# ── Blocking load ──────────────────────────────────────────────────────────────
def _load_model_blocking() -> None:
    global model
    download_from_gdrive(GDRIVE_FILE_ID, LOCAL_MODEL_PATH)

    if not os.path.exists(LOCAL_MODEL_PATH):
        raise RuntimeError(f"Model file missing after download: {LOCAL_MODEL_PATH}")

    size = os.path.getsize(LOCAL_MODEL_PATH)
    print(f"Loading model from {LOCAL_MODEL_PATH} (size={size/1e6:.1f} MB) ...")
    model = tf.keras.models.load_model(LOCAL_MODEL_PATH)
    print("Model loaded successfully.")


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _load_model_blocking)
    except Exception as exc:
        print(f"[STARTUP ERROR] Could not load model: {exc}")
    yield
    model = None


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Diabetic Foot Ulcer Classification API",
    description="EfficientNetB0-based binary classifier: Non_Ulcer vs Ulcer",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Response schema ────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: str
    probabilities: dict[str, str]


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image     = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image     = image.resize(IMAGE_SIZE, LANCZOS)
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Diabetic Foot Ulcer API is running",
        "classes": CLASS_NAMES,
        "input_size": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
        "model_ready": model is not None,
    }


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    if file.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG or PNG.",
        )

    image_bytes = await file.read()
    size_mb     = len(image_bytes) / 1e6
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum is {MAX_UPLOAD_MB} MB.",
        )

    try:
        img_array = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image processing error: {e}")

    try:
        with model_lock:
            raw_output = model.predict(img_array, verbose=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    probs           = raw_output[0]
    predicted_idx   = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence      = f"{round(float(probs[predicted_idx]) * 100, 2)}%"
    probabilities   = {
        name: f"{round(float(prob) * 100, 2)}%"
        for name, prob in zip(CLASS_NAMES, probs)
    }

    return PredictionResponse(
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probabilities,
    )
