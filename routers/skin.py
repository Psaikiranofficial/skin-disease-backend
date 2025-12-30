from fastapi import APIRouter, UploadFile, File
from PIL import Image
import tensorflow as tf
import numpy as np
import io
import os
import gdown

router = APIRouter(
    prefix="/skin",
    tags=["Skin Disease"]
)

# ===================== MODEL DOWNLOAD =====================

MODEL_PATH = "models/skin_disease_prediction_model.h5"
GDRIVE_FILE_ID = "1Qg5eMaFS3q8uncpj6iz3EzE_GzReFEQb"

if not os.path.exists(MODEL_PATH):
    os.makedirs("models", exist_ok=True)
    print("⬇️ Downloading skin disease model from Google Drive...")
    gdown.download(
        f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
        MODEL_PATH,
        quiet=False
    )

# ===================== LOAD MODEL =====================

model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (MUST match training order)
CLASS_NAMES = [
    "Acne and Rosacea Disease",
    "Atopic Dermatitis Disease",
    "Bacterial Skin Infection (Cellulitis / Impetigo)",
    "Eczema Disease",
    "Hair Loss (Alopecia)",
    "Contact Dermatitis (Poison Ivy)",
    "Psoriasis / Lichen Planus",
    "Benign Skin Tumors (Seborrheic Keratosis)",
    "Fungal Skin Infection (Ringworm / Candidiasis)",
    "Viral Skin Infection (Warts / Molluscum)"
]

# ===================== API =====================

@router.post("/predict")
async def predict_skin_disease(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image = image.resize((224, 224))

    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    class_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    return {
        "prediction": CLASS_NAMES[class_index],
        "confidence": round(confidence * 100, 2)
    }
