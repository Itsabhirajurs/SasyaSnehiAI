# PlantCare AI - Plant Disease Detection System

## Overview
A deep learning-powered web application that detects plant diseases using MobileNetV2 transfer learning. Built with Flask and trained on 87,000+ plant images across 38 disease/healthy classes.

## Quick Start

### 1. Start the Flask Server
```bash
cd filesss
python app.py
```

The app will be available at: **http://127.0.0.1:5000**

### 2. Use the App
- **Home**: View features and statistics (http://127.0.0.1:5000/)
- **About**: Learn about the technology (http://127.0.0.1:5000/about)
- **Upload**: Upload a plant image for disease detection (http://127.0.0.1:5000/upload)
- **Result**: View predictions with recommendations

## Training & Development Files
- `plant Disease Detection.ipynb` - Current end-to-end training notebook (Google Colab style) using `vipoooool/new-plant-diseases-dataset`.
- `train_better_model.py` - Improved local training script using EfficientNetB0 + two-stage fine-tuning.

### Train a Better Model (Kaggle Dataset)
1. Ensure your Kaggle API key is configured (place `kaggle.json` in `~/.kaggle/` or `%USERPROFILE%\\.kaggle\\`).
2. Run:

```bash
cd filesss
python train_better_model.py --dataset vipoooool/new-plant-diseases-dataset
```

Outputs are written to `model_artifacts/` (best model, final model, metrics, class names).

### Model loading priority in app
`app.py` now auto-loads models in this order:
1. `model_artifacts/efficientnet_best.keras`
2. `model_artifacts/efficientnet_final.keras`
3. `mobilenetv2_best.keras` (fallback)

If `model_artifacts/class_names.json` exists, those class names are used automatically.

### 3. Test with a Plant Image
1. Navigate to http://127.0.0.1:5000/upload
2. Click or drag-drop a plant leaf image
3. View results:
   - Plant type (Apple, Tomato, Grape, etc.)
   - Disease/health status
   - Confidence score
   - Treatment recommendations

## Project Structure
```
filesss/
├── app.py                          # Flask backend (routes, predictions)
├── mobilenetv2_best.keras          # Pre-trained model (10.9 MB)
├── plant Disease Detection.ipynb   # Training notebook for Google Colab
├── train_better_model.py           # Improved EfficientNetB0 training script
├── templates/
│   ├── home.html                   # Landing page
│   ├── upload.html                 # Image upload interface
│   ├── result.html                 # Prediction results display
│   └── about.html                  # Technology explanation
├── static/
│   ├── style.css                   # Responsive styling
│   └── images/                     # User uploaded images
└── uploads/                        # Temporary upload folder
```

## Model Architecture
- **Base**: MobileNetV2 (pre-trained on ImageNet)
- **Layers**: 
  - GlobalAveragePooling2D
  - Dropout(0.35)
  - Dense(256, ReLU)
  - Dropout(0.25)
  - Dense(38, Softmax) - 38 disease classes
- **Training**: 5 epochs, Adam optimizer (lr=1e-4)
- **Validation Accuracy**: 96.6%

## Supported Plants (38 Classes)
- Apple (3 diseases + healthy)
- Blueberry (healthy)
- Cherry (1 disease + healthy)
- Corn/Maize (3 diseases + healthy)
- Grape (3 diseases + healthy)
- Orange (1 disease)
- Peach (1 disease + healthy)
- Pepper Bell (1 disease + healthy)
- Potato (2 diseases + healthy)
- Raspberry (healthy)
- Soybean (healthy)
- Squash (1 disease)
- Strawberry (1 disease + healthy)
- Tomato (7 diseases + healthy)

## API Endpoints

### GET / 
Home page with features and statistics

### GET /about
About page with technology details

### GET /upload
Upload interface with drag-drop support

### POST /predict
Upload an image for disease prediction
- **Input**: multipart/form-data with 'file' field
- **Output**: JSON `{"success": true}` (prediction stored in session)
- **Redirect**: Browser redirects to /result

### GET /result
Display prediction results
- **Requires**: Active session with prediction data
- **Shows**: Plant type, condition, confidence, recommendations

## Requirements Met
✅ Exact implementation from PDF specification
✅ MobileNetV2 transfer learning architecture
✅ 38-class plant disease classification
✅ Flask web application
✅ Responsive HTML/CSS interface
✅ Drag-drop image upload
✅ Real-time predictions
✅ Session-based result storage
✅ Model gracefully handles when not available

## Browser Compatibility
- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge
- Requires JavaScript enabled

## Performance
- Model load time: ~4 seconds
- Prediction time: ~2 seconds per image
- Image preprocessing: <100ms
- Works on CPU (uses TensorFlow CPU mode)

## Notes
- Model file: `mobilenetv2_best.keras` (10.9 MB)
- Image input: 224x224 RGB
- Supports: JPG, PNG, BMP, GIF
- Maximum image size: Handled by browser file input
- Uploads are temporarily stored then deleted

---

Built with TensorFlow/Keras, Flask, and responsive web design.
