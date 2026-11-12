"""FastAPI application for weapon detection."""
import base64
import io
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

from api.config import (
    ALERT_IMAGES_DIR,
    ALERT_LOG_PATH,
    ALLOWED_EXTENSIONS,
    DEFAULT_CONFIDENCE,
    INFERENCE_IMG_SIZE,
    MAX_CONFIDENCE,
    MAX_UPLOAD_SIZE,
    MIN_CONFIDENCE,
    MODEL_PATH,
)
from api.schemas import Alert, AlertsResponse, Detection, HealthResponse, PredictResponse


# Global model instance
model: YOLO | None = None
device: str = "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, clean up at shutdown."""
    global model, device
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(MODEL_PATH))
    model.to(device)
    print(f"Model loaded on {device}: {MODEL_PATH}")
    yield
    model = None


app = FastAPI(
    title="Weapon Detection API",
    description="AI-based weapon detection for decision support. Not proof.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve alert images
app.mount("/alert-images", StaticFiles(directory=str(ALERT_IMAGES_DIR)), name="alert-images")


def validate_image_file(file: UploadFile) -> bytes:
    """Validate and read uploaded image file."""
    # Check extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    
    # Read file content
    content = file.file.read()
    
    # Check size
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_UPLOAD_SIZE // (1024*1024)} MB",
        )
    
    # Decode image safely
    try:
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {e}")
    
    return img


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model=str(MODEL_PATH),
        device=device,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(DEFAULT_CONFIDENCE, ge=MIN_CONFIDENCE, le=MAX_CONFIDENCE),
):
    """Run weapon detection on an uploaded image."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate and read image
    img = validate_image_file(file)
    
    # Run inference
    start = time.perf_counter()
    results = model.predict(img, imgsz=INFERENCE_IMG_SIZE, conf=conf, verbose=False)[0]
    inference_ms = (time.perf_counter() - start) * 1000
    
    # Extract detections
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(Detection(
            box=[float(x1), float(y1), float(x2), float(y2)],
            confidence=float(box.conf),
            class_name=results.names[int(box.cls)],
        ))
    
    # Create annotated image
    annotated = results.plot()
    _, buffer = cv2.imencode(".jpg", annotated)
    annotated_b64 = base64.b64encode(buffer).decode("utf-8")
    
    return PredictResponse(
        detections=detections,
        count=len(detections),
        threshold=conf,
        inference_ms=round(inference_ms, 2),
        annotated_image=annotated_b64,
    )


@app.get("/alerts", response_model=AlertsResponse)
async def get_alerts(limit: int = Query(20, ge=1, le=100)):
    """Get recent alerts from the log."""
    if not ALERT_LOG_PATH.exists():
        return AlertsResponse(alerts=[])
    
    import csv
    
    alerts = []
    with open(ALERT_LOG_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        for row in rows[-limit:]:
            alerts.append(Alert(
                timestamp=row["timestamp"],
                source=row["source"],
                frame=int(row["frame"]),
                video_time_s=float(row["video_time_s"]),
                max_conf=float(row["max_conf"]),
                evidence_image=row["evidence_image"],
            ))
    
    # Return newest first
    alerts.reverse()
    return AlertsResponse(alerts=alerts)


# Serve frontend at root
@app.get("/")
async def root():
    """Serve the frontend HTML page."""
    frontend_path = Path("frontend/index.html")
    if frontend_path.exists():
        return FileResponse(frontend_path)
    return {"message": "Frontend not built yet. See /docs for API."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)