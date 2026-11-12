"""Tests for the weapon detection API."""
import os
import tempfile
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from ultralytics import YOLO

from api.main import app
from api.config import MODEL_PATH


# Create a test client
client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def load_model():
    """Load model for tests."""
    from api.main import model, device
    import api.main as main_module
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main_module.model = YOLO(str(MODEL_PATH))
    main_module.model.to(device)
    main_module.device = device
    yield
    main_module.model = None


def test_health():
    """Test /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "device" in data
    assert data["device"] in ("cuda", "cpu")


def test_predict_valid_image():
    """Test /predict with a valid image."""
    # Use test.jpg from project root
    test_image = Path("test.jpg")
    if not test_image.exists():
        pytest.skip("test.jpg not found")
    
    with open(test_image, "rb") as f:
        response = client.post("/predict", files={"file": ("test.jpg", f, "image/jpeg")})
    
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert isinstance(data["detections"], list)
    assert "count" in data
    assert isinstance(data["count"], int)
    assert "threshold" in data
    assert isinstance(data["threshold"], float)
    assert "inference_ms" in data
    assert isinstance(data["inference_ms"], float)
    assert "annotated_image" in data
    assert isinstance(data["annotated_image"], str)
    assert len(data["annotated_image"]) > 0
    assert "disclaimer" in data
    assert data["disclaimer"] == "AI-generated detection. Not proof that a weapon is present. A human must review every alert."


def test_predict_invalid_file_type():
    """Test /predict with a text file renamed to .jpg."""
    # Create a temporary text file
    with tempfile.NamedTemporaryFile(suffix=".jpg", mode="wb", delete=False) as f:
        f.write(b"This is not an image")
        temp_path = f.name
    
    try:
        with open(temp_path, "rb") as f:
            response = client.post("/predict", files={"file": ("fake.jpg", f, "image/jpeg")})
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
    finally:
        os.unlink(temp_path)


def test_predict_file_too_large():
    """Test /predict with a file over 10 MB."""
    # Create a temporary file > 10 MB
    with tempfile.NamedTemporaryFile(suffix=".jpg", mode="wb", delete=False) as f:
        f.write(os.urandom(11 * 1024 * 1024))
        temp_path = f.name
    
    try:
        with open(temp_path, "rb") as f:
            response = client.post("/predict", files={"file": ("large.jpg", f, "image/jpeg")})
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "10 MB" in data["detail"] or "too large" in data["detail"].lower()
    finally:
        os.unlink(temp_path)


def test_alerts_endpoint():
    """Test /alerts endpoint."""
    response = client.get("/alerts?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "alerts" in data
    assert isinstance(data["alerts"], list)
    for alert in data["alerts"]:
        assert "timestamp" in alert
        assert "source" in alert
        assert "frame" in alert
        assert "video_time_s" in alert
        assert "max_conf" in alert
        assert "evidence_image" in alert