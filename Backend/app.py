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
import os
import zipfile
import gdown

# ---------- FASTAPI INIT ----------
app = FastAPI(title="DactyloAI Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "blood_fingerprint_FULL.csv"

# ---------- DOWNLOAD MODELS FROM GDRIVE ----------
def download_models():
    base_dir = Path("/tmp/models")
    base_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "efficientnet": {
            "id": "1NswGilnqR3vwRikPC0IXeLKBx47JsB7d",
            "zip": base_dir / "efficientnet.zip",
            "extract_to": base_dir / "efficientnet_savedmodel"
        },
        "inception": {
            "id": "1TVFTCfiJ2QnzuYjHqpfV8URhCq1-VeFy",
            "zip": base_dir / "inception.zip",
            "extract_to": base_dir / "inceptionv3_savedmodel"
        }
    }

    for name, info in models.items():
        if not info["extract_to"].exists():
            print(f"📥 Downloading {name}...")

            url = f"https://drive.google.com/uc?id={info['id']}"
            gdown.download(url, str(info["zip"]), quiet=False)

            print(f"📦 Extracting {name}...")
            with zipfile.ZipFile(info["zip"], 'r') as zip_ref:
                zip_ref.extractall(info["extract_to"])

    return base_dir

# ✅ IMPORTANT: actually call it
models_dir = download_models()

# ---------- LOAD MODELS ----------
def load_savedmodel_safe(path: Path, model_name: str):
    if not path.exists():
        print(f"❌ Model not found: {path}")
        return None

    try:
        print(f"🔄 Loading {model_name}...")
        model = tf.saved_model.load(str(path))
        infer = model.signatures["serving_default"]
        print(f"✅ {model_name} loaded")
        return infer
    except Exception as e:
        print(f"❌ Failed loading {model_name}: {e}")
        traceback.print_exc()
        return None

print("🚀 Loading models...")

efficient_model_path = models_dir / "efficientnet_savedmodel"
inception_model_path = models_dir / "inceptionv3_savedmodel"
blood_model_path = models_dir / "model_blood_group_detection.keras"

efficient_model = load_savedmodel_safe(efficient_model_path, "EfficientNet")
inception_model = load_savedmodel_safe(inception_model_path, "InceptionV3")

# Blood model (optional)
if blood_model_path.exists():
    try:
        blood_model = tf.keras.models.load_model(str(blood_model_path), compile=False)
        print("✅ Blood model loaded")
    except:
        blood_model = None
        print("❌ Blood model failed")
else:
    blood_model = None

# ---------- LABELS ----------
PATTERN_TYPES = ["class1_arc", "class2_whorl", "class3_loop"]
BLOOD_TYPES = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]
ENSEMBLE_WEIGHTS = {'efficientnet': 0.55, 'inception': 0.45}

# ---------- PREPROCESS ----------
def preprocess_image(img_bytes, target_size):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = np.array(img)

    img = cv2.resize(img, (target_size[1], target_size[0]))
    img = np.expand_dims(img.astype(np.float32), axis=0)

    return img

# ---------- FINGERPRINT ----------
@app.post("/predict/fingerprint")
async def predict_fingerprint(file: UploadFile = File(...)):
    if not (efficient_model or inception_model):
        return JSONResponse(status_code=503, content={"error": "No models loaded"})

    try:
        img_bytes = await file.read()
        img = preprocess_image(img_bytes, (224, 224))
        img_tensor = tf.convert_to_tensor(img)

        preds = []
        weights = []

        if inception_model:
            out = inception_model(input_layer_3=img_tensor)
            key = list(out.keys())[0]
            preds.append(out[key].numpy())
            weights.append(ENSEMBLE_WEIGHTS["inception"])

        if efficient_model:
            out = efficient_model(input_layer_2=img_tensor)
            key = list(out.keys())[0]
            preds.append(out[key].numpy())
            weights.append(ENSEMBLE_WEIGHTS["efficientnet"])

        # Ensemble
        if len(preds) > 1:
            total = sum(weights)
            final = sum(p * (w / total) for p, w in zip(preds, weights))[0]
        else:
            final = preds[0][0]

        idx = int(np.argmax(final))

        return {
            "pattern": PATTERN_TYPES[idx],
            "confidence": float(final[idx])
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------- BLOOD ----------
@app.post("/predict/blood")
async def predict_blood(file: UploadFile = File(...)):
    if blood_model is None:
        return JSONResponse(status_code=503, content={"error": "Blood model not loaded"})

    try:
        img_bytes = await file.read()
        img = preprocess_image(img_bytes, (256, 256))
        img = resnet_preprocess(img)

        pred = blood_model.predict(img)
        idx = int(np.argmax(pred[0]))

        return {
            "blood_type": BLOOD_TYPES[idx],
            "confidence": float(pred[0][idx])
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------- HEALTH ----------
@app.get("/")
def health():
    return {
        "status": "running",
        "models": {
            "efficientnet": efficient_model is not None,
            "inception": inception_model is not None,
            "blood": blood_model is not None
        }
    }

# ---------- ENTRY ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)