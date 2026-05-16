import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import math
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml_models", "model_tabungan_production.keras")
SCALER_PATH = os.path.join(BASE_DIR, "ml_models", "scaler_x_tabungan.pkl")

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict_estimasi_tabungan(terkumpul: float, target: float, nabung: float) -> dict:
    input_data = pd.DataFrame({
        'total_terkumpul': [np.log1p(terkumpul)],
        'target_nominal': [np.log1p(target)],
        'nominal_nabung': [np.log1p(nabung)]
    })

    input_scaled = scaler.transform(input_data)

    res_scaled = model.predict(input_scaled, verbose=0)

    prediksi_raw = np.expm1(res_scaled).flatten()[0]

    estimasi_kali_nabung = math.ceil(prediksi_raw)

    return {
        "prediksi_raw": float(prediksi_raw),
        "estimasi_kali_nabung": int(estimasi_kali_nabung)
    }