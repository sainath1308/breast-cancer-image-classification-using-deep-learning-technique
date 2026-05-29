from keras.layers import Dense

# 🔥 PATCH: Ignore quantization_config
original_dense_init = Dense.__init__

def new_dense_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    original_dense_init(self, *args, **kwargs)

Dense.__init__ = new_dense_init

import os
import io
import logging
import numpy as np
from PIL import Image
import tensorflow as tf

logger = logging.getLogger(__name__)

IMG_SIZE = (224, 224)
DEFAULT_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))

_models = {}

# ✅ LOAD MODEL
def load_model():
    global _models

    if _models:
        return _models

    model_path = os.path.join(os.path.dirname(__file__), "saved_model", "densenet_final.keras")

    if os.path.exists(model_path):
        _models["single"] = tf.keras.models.load_model(model_path, compile=False)
        print("✅ Model loaded successfully")
    else:
        print("❌ Model not found")

    return _models


# ✅ PREPROCESS IMAGE (FIXED)
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)

    # 🔥 IMPORTANT FIX (NO /255)
    arr = np.array(img, dtype=np.float32)

    arr = np.expand_dims(arr, axis=0)
    return arr


# ✅ PREDICT FUNCTION
def predict(image_bytes: bytes, include_heatmap: bool = False, threshold: float = DEFAULT_THRESHOLD) -> dict:
    
    models = load_model()

    if not models:
        return {
            "error": "No trained models loaded.",
            "prediction": None,
            "confidence": None,
        }

    input_tensor = preprocess_image(image_bytes)

    model = list(models.values())[0]

    # ✅ convert to float
    pred = float(model.predict(input_tensor, verbose=0)[0][0])

    idc_positive_prob = float(pred)
    idc_negative_prob = float(1 - pred)

    is_malignant = pred >= threshold
    label = "Malignant" if is_malignant else "Benign"

    confidence = float(idc_positive_prob if is_malignant else idc_negative_prob)

    # ✅ RESPONSE
    response = {
        "prediction": label,
        "confidence": round(confidence, 4),
        "idc_positive_prob": round(idc_positive_prob, 4),
        "idc_negative_prob": round(idc_negative_prob, 4),
        "is_malignant": is_malignant,
        "img_size_used": f"{IMG_SIZE[0]}x{IMG_SIZE[1]}",
        "threshold_used": round(float(threshold), 3),
        "model_info": "DenseNet121"
    }

    return response


# ✅ MODEL INFO
def get_model_info() -> dict:
    models = load_model()

    if not models:
        return {"status": "not_loaded"}

    info = {}

    for name, model in models.items():
        info[name] = {
            "architecture": model.name,
            "input_shape": list(model.input_shape),
            "total_params": model.count_params(),
        }

    return {
        "status": "loaded",
        "models_loaded": list(models.keys()),
        "model_details": info,
        "classes": ["Benign", "Malignant"]
    }