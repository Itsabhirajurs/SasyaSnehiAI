"""Central configuration for Sashyasnehi AI MVP."""

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
STATIC_FOLDER = BASE_DIR / "static"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "disease_model.keras"
LEGACY_MODEL_PATH = BASE_DIR / "mobilenetv2_best.keras"
CLASS_NAMES_PATH = BASE_DIR / "model_artifacts" / "class_names.json"
IMAGE_SIZE = (224, 224)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
