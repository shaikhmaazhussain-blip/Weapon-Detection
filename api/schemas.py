"""Pydantic models for API requests and responses."""
from typing import List, Optional
from pydantic import BaseModel, Field


class Detection(BaseModel):
    """Single detection result."""
    box: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., description="Detection confidence score")
    class_name: str = Field(..., description="Class name (e.g., 'weapon')")


class PredictResponse(BaseModel):
    """Response for /predict endpoint."""
    detections: List[Detection]
    count: int
    threshold: float
    inference_ms: float
    annotated_image: str = Field(..., description="Base64 encoded JPEG of annotated image")
    disclaimer: str = "AI-generated detection. Not proof that a weapon is present. A human must review every alert."


class Alert(BaseModel):
    """Single alert entry from the alert log."""
    timestamp: str
    source: str
    frame: int
    video_time_s: float
    max_conf: float
    evidence_image: str


class AlertsResponse(BaseModel):
    """Response for /alerts endpoint."""
    alerts: List[Alert]


class HealthResponse(BaseModel):
    """Response for /health endpoint."""
    status: str
    model: str
    device: str