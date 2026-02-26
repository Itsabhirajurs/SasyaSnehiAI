"""Model loading and prediction utilities for disease classification."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

from config import CLASS_NAMES_PATH, IMAGE_SIZE, LEGACY_MODEL_PATH, MODEL_PATH


DEFAULT_CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


class DiseaseModelLoader:
    """Load a Keras disease model once and perform predictions."""

    def __init__(self):
        self.model = None
        self.model_available = False
        self.model_path_in_use = None
        self.class_names = self._load_class_names()

    def _load_class_names(self):
        path = Path(CLASS_NAMES_PATH)
        if not path.exists():
            return DEFAULT_CLASS_NAMES
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            if isinstance(data, list) and data:
                return data
        except Exception:
            pass
        return DEFAULT_CLASS_NAMES

    def load_once(self):
        if self.model is not None:
            return

        candidates = [Path(MODEL_PATH), Path(LEGACY_MODEL_PATH)]
        selected = next((candidate for candidate in candidates if candidate.exists()), None)

        if not selected:
            self.model_available = False
            return

        self.model = load_model(selected)
        self.model_path_in_use = str(selected)
        self.model_available = True

    def _format_label(self, label):
        if "___" in label:
            plant, condition = label.split("___", 1)
        else:
            plant, condition = label, "unknown"
        return plant.replace("_", " "), condition.replace("_", " ")

    def predict(self, image_path):
        self.load_once()
        if not self.model_available:
            return {
                "label": "Model unavailable",
                "plant_type": "Unknown",
                "condition": "Model file missing",
                "confidence": 0.0,
                "status": "unavailable",
                "recommendations": [
                    "Add model/disease_model.keras to enable AI prediction",
                    "Until then, use visual inspection or expert consultation",
                    "Capture clear leaf images in good lighting",
                ],
            }

        image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
        array = np.array(image, dtype=np.float32)
        array = np.expand_dims(array, axis=0)
        array = preprocess_input(array)

        predictions = self.model.predict(array)
        index = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions))
        label = self.class_names[index] if index < len(self.class_names) else "unknown"
        plant_type, condition = self._format_label(label)

        if "healthy" in condition.lower():
            status = "healthy"
            recommendations = [
                "Continue regular watering and care",
                "Ensure adequate sunlight and balanced nutrients",
                "Monitor leaves weekly for early signs",
            ]
        else:
            status = "diseased"
            recommendations = [
                "Isolate affected leaves and avoid overhead irrigation",
                "Start targeted treatment as advised",
                "Monitor nearby plants for spread",
            ]

        return {
            "label": label,
            "plant_type": plant_type,
            "condition": condition,
            "confidence": confidence,
            "status": status,
            "recommendations": recommendations,
        }
