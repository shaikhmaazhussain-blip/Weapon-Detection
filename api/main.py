"""FastAPI application for weapon detection."""
import base64
import csv
import io
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
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
from api.schemas import Alert, AlertLogResponse, AlertsResponse, Detection, HealthResponse, PredictResponse


# Global model instance
model: YOLO | None = None
device: str = "cpu"

# Alert logging rate limit
_last_alert_time: float = 0.0
_alert_lock = Lock()
_alert_counter = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, clean up at shutdown."""
    global model, device
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(MODEL_PATH))
    model.to(device)
    print(f"Model loaded on {device}: {MODEL_PATH}")
    
    # Warm up: run one dummy inference to avoid cold start
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy_img, imgsz=INFERENCE_IMG_SIZE, conf=DEFAULT_CONFIDENCE, verbose=False)
    print("Model warmed up with dummy inference")
    
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


def validate_image_file(file: UploadFile) -> np.ndarray:
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


@app.post("/alerts/log", response_model=AlertLogResponse)
async def log_alert(
    file: UploadFile = File(...),
    max_conf: float = Form(...),
    source: str = Form(...),
):
    """Log an alert with evidence image."""
    global _last_alert_time, _alert_counter

    # Rate limit: max 1 alert per 2 seconds
    with _alert_lock:
        now = time.time()
        if now - _last_alert_time < 2.0:
            raise HTTPException(
                status_code=429,
                detail="Rate limited: maximum one alert per 2 seconds",
            )
        _last_alert_time = now
        _alert_counter += 1
        counter = _alert_counter

    # Validate and read image
    img = validate_image_file(file)

    # Generate unique filename
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"alert_{timestamp_str}_{counter}.jpg"
    filepath = ALERT_IMAGES_DIR / filename

    # Save evidence image
    cv2.imwrite(str(filepath), img)

    # Prepare log row (same columns as alert_video.py)
    timestamp = datetime.now().isoformat()
    frame = 0  # Not applicable for API
    video_time_s = 0.0  # Not applicable for API
    evidence_image = f"alerts/images/{filename}"

    # Append to CSV (write header if new)
    file_exists = ALERT_LOG_PATH.exists()
    with open(ALERT_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "source", "frame", "video_time_s", "max_conf", "evidence_image"])
        writer.writerow([timestamp, source, frame, video_time_s, max_conf, evidence_image])

    return AlertLogResponse(
        timestamp=timestamp,
        source=source,
        frame=frame,
        video_time_s=video_time_s,
        max_conf=max_conf,
        evidence_image=evidence_image,
        url=f"/alert-images/{filename}",
    )


# Serve frontend at root
@app.get("/")
async def root():
    """Serve the frontend HTML page."""
    frontend_path = Path("frontend/index.html")
    if frontend_path.exists():
        return FileResponse(frontend_path)
    return {"message": "Frontend not built yet. See /docs for API."}


@app.get("/legal")
async def legal():
    """Serve the legal HTML page."""
    legal_path = Path("frontend/legal.html")
    if legal_path.exists():
        return FileResponse(legal_path)
    raise HTTPException(status_code=404, detail="Legal page not found")


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)