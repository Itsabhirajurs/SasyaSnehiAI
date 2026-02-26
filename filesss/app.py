import os
import secrets
import json

import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, "uploads")
STATIC_FOLDER = os.path.join(APP_ROOT, "static")
IMAGE_SIZE = (224, 224)

MODEL_CANDIDATES = [
    {
        "model_type": "efficientnet",
        "model_path": os.path.join(APP_ROOT, "model_artifacts", "efficientnet_best.keras"),
        "class_names_path": os.path.join(APP_ROOT, "model_artifacts", "class_names.json"),
    },
    {
        "model_type": "efficientnet",
        "model_path": os.path.join(APP_ROOT, "model_artifacts", "efficientnet_final.keras"),
        "class_names_path": os.path.join(APP_ROOT, "model_artifacts", "class_names.json"),
    },
    {
        "model_type": "mobilenet",
        "model_path": os.path.join(APP_ROOT, "mobilenetv2_best.keras"),
        "class_names_path": None,
    },
]

CLASS_NAMES = [
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


app = Flask(__name__)
app.secret_key = "plantcare_ai"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["STATIC_FOLDER"] = STATIC_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, "images"), exist_ok=True)


model = None
MODEL_AVAILABLE = False
MODEL_TYPE = None
MODEL_PATH_IN_USE = None
CLASS_NAMES_IN_USE = CLASS_NAMES


def load_class_names(class_names_path):
    if not class_names_path or not os.path.exists(class_names_path):
        return CLASS_NAMES
    try:
        with open(class_names_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, list) and loaded:
            return loaded
        if isinstance(loaded, dict) and loaded:
            if all(isinstance(v, int) for v in loaded.values()):
                ordered = [None] * (max(loaded.values()) + 1)
                for class_name, class_id in loaded.items():
                    if 0 <= class_id < len(ordered):
                        ordered[class_id] = class_name
                if all(item is not None for item in ordered):
                    return ordered
            if all(str(k).isdigit() for k in loaded.keys()):
                indexed = sorted(((int(k), v) for k, v in loaded.items()), key=lambda x: x[0])
                return [v for _, v in indexed]
    except Exception:
        pass
    return CLASS_NAMES


def load_model_once():
    global model, MODEL_AVAILABLE, MODEL_TYPE, MODEL_PATH_IN_USE, CLASS_NAMES_IN_USE
    if model is None:
        selected = next(
            (candidate for candidate in MODEL_CANDIDATES if os.path.exists(candidate["model_path"])),
            None,
        )
        if not selected:
            MODEL_AVAILABLE = False
            print("No model file found. Running in fallback mode without AI model.")
            return

        MODEL_TYPE = selected["model_type"]
        MODEL_PATH_IN_USE = selected["model_path"]
        CLASS_NAMES_IN_USE = load_class_names(selected["class_names_path"])

        model = load_model(MODEL_PATH_IN_USE)
        MODEL_AVAILABLE = True
        print(f"Model loaded successfully from: {MODEL_PATH_IN_USE}")
        print(f"Model type: {MODEL_TYPE}")
        print(f"Class names loaded: {len(CLASS_NAMES_IN_USE)}")


def format_label(label):
    if "___" in label:
        plant, condition = label.split("___", 1)
    else:
        plant, condition = label, "unknown"
    plant = plant.replace("_", " ")
    condition = condition.replace("_", " ")
    return plant, condition


def predict_image(image_path):
    load_model_once()

    if not MODEL_AVAILABLE:
        return {
            "label": "Model unavailable",
            "plant_type": "Unknown",
            "condition": "Model file missing",
            "confidence": 0.0,
            "status": "unavailable",
            "recommendations": [
                "Add mobilenetv2_best.keras to enable AI prediction",
                "Until then, use visual inspection or expert consultation",
                "Capture clear leaf images in good lighting",
                "Re-run analysis after adding the model file",
            ],
        }

    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    if MODEL_TYPE == "mobilenet":
        arr = mobilenet_preprocess(arr)

    preds = model.predict(arr)
    idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    label = CLASS_NAMES_IN_USE[idx] if idx < len(CLASS_NAMES_IN_USE) else "unknown"
    plant_type, condition = format_label(label)

    if "healthy" in condition.lower():
        status = "healthy"
        recommendations = [
            "Continue regular watering and care",
            "Ensure adequate sunlight and nutrients",
            "Monitor for any changes in appearance",
            "Maintain good air circulation",
        ]
    else:
        status = "diseased"
        recommendations = [
            "Isolate affected plants to prevent spread",
            "Consult with an agricultural expert",
            "Consider appropriate treatment methods",
            "Monitor other plants for similar symptoms",
        ]

    return {
        "label": label,
        "plant_type": plant_type,
        "condition": condition,
        "confidence": confidence,
        "status": status,
        "recommendations": recommendations,
    }


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/upload")
def upload():
    return render_template("upload.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            prediction = predict_image(filepath)
            static_filename = f"upload_{secrets.token_hex(8)}.jpg"
            static_path = os.path.join(
                app.config["STATIC_FOLDER"], "images", static_filename
            )
            Image.open(filepath).save(static_path)

            session["prediction"] = prediction
            session["image_path"] = f"images/{static_filename}"

            os.remove(filepath)
            return jsonify({"success": True})
        except Exception as exc:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": str(exc)}), 500


@app.route("/result")
def result():
    prediction = session.get("prediction")
    image_path = session.get("image_path")

    if not prediction:
        return redirect(url_for("upload"))

    return render_template(
        "result.html", prediction=prediction, image_path=image_path
    )


if __name__ == "__main__":
    load_model_once()
    app.run(debug=True, host="0.0.0.0", port=5000)
