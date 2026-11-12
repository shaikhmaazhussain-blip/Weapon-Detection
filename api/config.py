"""Configuration settings for the weapon detection API."""
import os
from pathlib import Path

# Model settings
MODEL_PATH = Path("runs/detect/v3/weights/best.pt")
DEFAULT_CONFIDENCE = 0.35
MIN_CONFIDENCE = 0.05
MAX_CONFIDENCE = 0.95

# Image settings
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
INFERENCE_IMG_SIZE = 640

# Alerts
ALERTS_DIR = Path("alerts")
ALERT_IMAGES_DIR = ALERTS_DIR / "images"
ALERT_LOG_PATH = ALERTS_DIR / "alert_log.csv"

# Device (will be set at runtime)
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu"