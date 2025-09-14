import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -------------------------------
# Paths
# -------------------------------
MODEL_PATH = "models/plant_disease_model.h5"
DATA_PATH = "data/disease_remedies.json"
DEFAULT_IMAGE_SIZE = (200, 200)  # match original Gradio imgSize

# -------------------------------
# Labels mapping from original Gradio app
# -------------------------------
labels = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Not a plant',
    5: 'Blueberry___healthy',
    6: 'Cherry___Powdery_mildew',
    7: 'Cherry___healthy',
    8: 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    9: 'Corn___Common_rust',
    10: 'Corn___Northern_Leaf_Blight',
    11: 'Corn___healthy',
    12: 'Grape___Black_rot',
    13: 'Grape___Esca_(Black_Measles)',
    14: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    15: 'Grape___healthy',
    16: 'Orange___Haunglongbing_(Citrus_greening)',
    17: 'Peach___Bacterial_spot',
    18: 'Peach___healthy',
    19: 'Pepper,_bell___Bacterial_spot',
    20: 'Pepper,_bell___healthy',
    21: 'Potato___Early_blight',
    22: 'Potato___Late_blight',
    23: 'Potato___healthy',
    24: 'Raspberry___healthy',
    25: 'Soybean___healthy',
    26: 'Squash___Powdery_mildew',
    27: 'Strawberry___Leaf_scorch',
    28: 'Strawberry___healthy',
    29: 'Tomato___Bacterial_spot',
    30: 'Tomato___Early_blight',
    31: 'Tomato___Late_blight',
    32: 'Tomato___Leaf_Mold',
    33: 'Tomato___Septoria_leaf_spot',
    34: 'Tomato___Spider_mites Two-spotted_spider_mite',
    35: 'Tomato___Target_Spot',
    36: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    37: 'Tomato___Tomato_mosaic_virus',
    38: 'Tomato___healthy'
}

CLASSES = [labels[i] for i in range(len(labels))]

# -------------------------------
# Load model
# -------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
_MODEL = load_model(MODEL_PATH)

# -------------------------------
# Load remedies JSON
# -------------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Disease remedies file not found at {DATA_PATH}")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    REMEDIES = json.load(f)

# -------------------------------
# Preprocess image
# -------------------------------
def _preprocess_image(img_path, target_size=None):
    if target_size is None:
        target_size = DEFAULT_IMAGE_SIZE
    img = image.load_img(img_path, target_size=target_size)
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# -------------------------------
# Predict disease
# -------------------------------
def predict_disease(image_path: str, lang="en"):
    try:
        img_arr = _preprocess_image(image_path, target_size=DEFAULT_IMAGE_SIZE)
        preds = _MODEL.predict(img_arr)
        if preds.ndim == 2:
            probs = preds[0]
        else:
            probs = preds.flatten()[:len(CLASSES)]

        predicted_idx = int(np.argmax(probs))
        confidence = float(np.max(probs))
        disease_name = CLASSES[predicted_idx] if predicted_idx < len(CLASSES) else f"Class_{predicted_idx}"

        # Lookup remedies
        item = REMEDIES.get(str(predicted_idx))
        if item and item.get("disease") == disease_name:
            crop = item.get("crop", "Unknown")
            remedy_en = item.get("remedy", {}).get("en", "")
            precautions_en = item.get("precautions", {}).get("en", "")
            remedy_hi = item.get("remedy", {}).get("hi", "")
            precautions_hi = item.get("precautions", {}).get("hi", "")
            remedy_ta = item.get("remedy", {}).get("ta", "")
            precautions_ta = item.get("precautions", {}).get("ta", "")
            remedy_te = item.get("remedy", {}).get("te", "")
            precautions_te = item.get("precautions", {}).get("te", "")
            remedy_ml = item.get("remedy", {}).get("ml", "")
            precautions_ml = item.get("precautions", {}).get("ml", "")

            # Choose remedy by lang
            if lang == "hi" and remedy_hi:
                remedy = remedy_hi
                precautions = precautions_hi or precautions_en
            elif lang == "ta" and remedy_ta:
                remedy = remedy_ta
                precautions = precautions_ta or precautions_en
            elif lang == "te" and remedy_te:
                remedy = remedy_te
                precautions = precautions_te or precautions_en
            elif lang == "ml" and remedy_ml:
                remedy = remedy_ml
                precautions = precautions_ml or precautions_en
            else:
                remedy = remedy_en
                precautions = precautions_en
        else:
            crop = "Unknown"
            remedy_en = remedy_hi = remedy_ta = remedy_te = remedy_ml = ""
            precautions_en = precautions_hi = precautions_ta = precautions_te = precautions_ml = ""
            remedy = "No remedy found in database."
            precautions = "No precautions available."

        return {
            "crop": crop,
            "disease": disease_name,
            "remedy": remedy,
            "precautions": precautions,
            "remedy_en": remedy_en,
            "precautions_en": precautions_en,
            "remedy_hi": remedy_hi,
            "precautions_hi": precautions_hi,
            "remedy_ta": remedy_ta,
            "precautions_ta": precautions_ta,
            "remedy_te": remedy_te,
            "precautions_te": precautions_te,
            "remedy_ml": remedy_ml,
            "precautions_ml": precautions_ml,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        return {
            "crop": "Unknown",
            "disease": "Error",
            "remedy": "",
            "precautions": f"Error processing image: {e}",
            "remedy_en": "",
            "precautions_en": "",
            "remedy_hi": "",
            "precautions_hi": "",
            "remedy_ta": "",
            "precautions_ta": "",
            "remedy_te": "",
            "precautions_te": "",
            "remedy_ml": "",
            "precautions_ml": "",
            "confidence": 0.0
        }