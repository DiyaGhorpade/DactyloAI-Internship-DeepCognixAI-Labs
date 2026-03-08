from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from pathlib import Path
from PIL import Image
import numpy as np
import io
import uvicorn
import cv2
import tensorflow as tf
import traceback
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  
import base64
from io import BytesIO
from scipy.stats import chi2_contingency

app = FastAPI(title="DactyloAI Backend API")

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
models_dir = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "blood_fingerprint_FULL.csv"

# ---------- LOAD SAVEDMODEL FORMAT (Cross-version compatible) ----------
def load_savedmodel_safe(path: Path, model_name: str):
    """Load SavedModel format (works across TF versions)"""
    if not path.exists():
        print(f"❌ Model directory not found: {path}")
        return None

    try:
        print(f"🔄 Loading {model_name} from {path}...")
        model = tf.saved_model.load(str(path))
        # Get the inference function
        infer = model.signatures["serving_default"]
        print(f"✅ {model_name} loaded successfully")
        return infer
    except Exception as e:
        print(f"❌ Failed loading {model_name}: {e}")
        traceback.print_exc()
        return None

# ---------- LOAD MODELS ----------
print("🚀 Starting model loading...")

# SavedModel paths (unzipped directories)
efficient_model_path = models_dir / "efficientnet_savedmodel"
inception_model_path = models_dir / "inceptionv3_savedmodel"
blood_model_path = models_dir / "model_blood_group_detection.keras"

# Load fingerprint models
efficient_model = load_savedmodel_safe(efficient_model_path, "EfficientNet")
inception_model = load_savedmodel_safe(inception_model_path, "InceptionV3")

# Load blood model (assuming it's compatible)
if blood_model_path.exists():
    try:
        blood_model = tf.keras.models.load_model(str(blood_model_path), compile=False)
        print("✅ Blood model loaded")
    except:
        blood_model = None
        print("❌ Blood model failed to load")
else:
    blood_model = None

# Check if models loaded
if efficient_model is None and inception_model is None:
    print("💥 WARNING: No fingerprint models were loaded successfully!")

# --- Define Class Labels ---
PATTERN_TYPES = ["class1_arc", "class2_whorl", "class3_loop"]
BLOOD_TYPES = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]
ENSEMBLE_WEIGHTS = {'efficientnet': 0.55, 'inception': 0.45}

# ---------- PREPROCESSING ----------
def preprocess_image_for_prediction(img_bytes: bytes, target_size: tuple) -> np.ndarray:
    """Preprocess image for prediction"""
    try:
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img)

        # Ensure 3 channels
        if img_array.ndim == 2:
            img_array = np.stack([img_array]*3, axis=-1)
        elif img_array.shape[-1] == 1:
            img_array = np.concatenate([img_array]*3, axis=-1)
        elif img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]

        # Resize
        target_size_int = (int(target_size[1]), int(target_size[0]))
        img_array_resized = cv2.resize(img_array, target_size_int, interpolation=cv2.INTER_AREA)
        
        # Convert to float32 and add batch dimension
        img_batch = np.expand_dims(img_array_resized.astype(np.float32), axis=0)
        return img_batch

    except Exception as e:
        print(f"❌ Image preprocessing failed: {e}")
        raise ValueError(f"Failed to preprocess image: {e}")




# ---------- FINGERPRINT PREDICTION ----------
@app.post("/predict/fingerprint")
async def predict_fingerprint(file: UploadFile = File(...)):
    available_models = [m for m in [inception_model, efficient_model] if m is not None]
    if not available_models:
        return JSONResponse(status_code=503, content={"error": "No fingerprint models available"})

    try:
        print(f"🎯 Received fingerprint prediction request: {file.filename}")
        img_bytes = await file.read()

        # Preprocess image
        img_batch = preprocess_image_for_prediction(img_bytes, target_size=(224, 224))
        
        # Convert to TensorFlow tensor
        img_tensor = tf.convert_to_tensor(img_batch, dtype=tf.float32)

        predictions = []
        model_weights = []
        per_model_confidences = {}

        # Run predictions
        # Run predictions with CORRECT input names
        if inception_model is not None:
            try:
                print("🔄 Running Inception prediction...")
                pred_inc = inception_model(input_layer_3=img_tensor)  # ✅ CORRECT
                output_key = list(pred_inc.keys())[0]
                pred_inc_array = pred_inc[output_key].numpy()
                predictions.append(pred_inc_array)
                model_weights.append(ENSEMBLE_WEIGHTS['inception'])
                per_model_confidences["inception"] = float(np.max(pred_inc_array[0]))
            except Exception as e:
                print(f"⚠️ Inception prediction failed: {e}")
                traceback.print_exc()

        if efficient_model is not None:
            try:
                print("🔄 Running EfficientNet prediction...")
                pred_eff = efficient_model(input_layer_2=img_tensor)  # ✅ CORRECT
                output_key = list(pred_eff.keys())[0]
                pred_eff_array = pred_eff[output_key].numpy()
                predictions.append(pred_eff_array)
                model_weights.append(ENSEMBLE_WEIGHTS['efficientnet'])
                per_model_confidences["efficientnet"] = float(np.max(pred_eff_array[0]))
            except Exception as e:
                print(f"⚠️ EfficientNet prediction failed: {e}")
                traceback.print_exc()


        if not predictions:
            return JSONResponse(status_code=500, content={"error": "All predictions failed"})

        # Weighted ensemble
        if len(predictions) > 1:
            total_weight = sum(model_weights)
            normalized_weights = [w / total_weight for w in model_weights]
            ensemble = sum(pred * weight for pred, weight in zip(predictions, normalized_weights))
            final_probs = ensemble[0]
            model_used_str = "weighted_ensemble"
        else:
            final_probs = predictions[0][0]
            model_used_str = "inception" if inception_model else "efficientnet"

        # Get final prediction
        idx = int(np.argmax(final_probs))
        confidence = float(final_probs[idx])
        pattern = PATTERN_TYPES[idx]

        print(f"🎯 Final prediction: {pattern} (confidence: {confidence:.4f})")

        return {
            "pattern": pattern,
            "class": idx,
            "confidence": confidence,
            "model_used": model_used_str,
            "probabilities": {
                PATTERN_TYPES[i]: float(final_probs[i])
                for i in range(len(PATTERN_TYPES))
            },
            "per_model_confidence": per_model_confidences,
        }

    except Exception as e:
        print(f"💥 Fingerprint prediction error: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------- BLOOD PREDICTION ----------
@app.post("/predict/blood")
async def predict_blood(file: UploadFile = File(...)):
    if blood_model is None:
        return JSONResponse(status_code=503, content={"error": "Blood model not loaded"})

    try:
        img_bytes = await file.read()
        img_batch = preprocess_image_for_prediction(img_bytes, target_size=(256, 256))
        x_blood = resnet_preprocess(img_batch)
        pred = blood_model.predict(x_blood, verbose=0)
        final_probs = pred[0]
        idx = int(np.argmax(final_probs))
        confidence = float(final_probs[idx])
        blood_type = BLOOD_TYPES[idx]

        return {
            "blood_type": blood_type,
            "confidence": confidence,
            "probabilities": {
                BLOOD_TYPES[i]: float(final_probs[i])
                for i in range(len(BLOOD_TYPES))
            },
        }

    except Exception as e:
        print(f"💥 Blood prediction erro+r: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
# ---------------------- ANALYTICS API ----------------------
from fastapi.staticfiles import StaticFiles
from services.analytics_engine import run_analytics

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/analytics")
def analytics():
    return run_analytics()

     
# ---------- HEALTH CHECK ----------
@app.get("/")
async def health_check():
    return {
        "status": "API is running",
        "models_loaded": {
            "inception": inception_model is not None,
            "efficientnet": efficient_model is not None,
            "blood": blood_model is not None
        }
    }

if __name__ == "__main__":
    print("🚀 Starting DactyloAI FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
