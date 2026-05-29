import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import math
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml_models", "model_lstm_tabungan.keras")
SCALER_X_PATH = os.path.join(BASE_DIR, "ml_models", "scaler_lstm.pkl")
SCALER_Y_PATH = os.path.join(BASE_DIR, "ml_models", "scaler_y_lstm.pkl")

model = tf.keras.models.load_model(MODEL_PATH)
scaler_x = joblib.load(SCALER_X_PATH)
scaler_y = joblib.load(SCALER_Y_PATH)

SEQ_LENGTH = 5

def predict_estimasi_tabungan(riwayat: list) -> dict:
    data_histori = pd.DataFrame([r.model_dump() for r in riwayat])
    
    data_histori['sisa_nominal'] = data_histori['target_nominal'] - data_histori['total_terkumpul']
    data_histori['rumus_kalkulator'] = np.ceil(data_histori['sisa_nominal'] / (data_histori['nominal_nabung'] + 1))
    
    fitur_x = [
        'target_nominal', 'nominal_nabung', 'total_terkumpul', 
        'jarak_hari_nabung', 'sisa_nominal', 'rumus_kalkulator'
    ]
    data_histori = data_histori[fitur_x]

    riwayat_scaled = scaler_x.transform(data_histori)

    jumlah_data = len(riwayat_scaled)
    if jumlah_data < SEQ_LENGTH:
        jumlah_padding = SEQ_LENGTH - jumlah_data
        padding = np.zeros((jumlah_padding, len(fitur_x)))
        riwayat_final = np.vstack([padding, riwayat_scaled])
    else:
        riwayat_final = riwayat_scaled[-SEQ_LENGTH:]

    input_tensor = np.expand_dims(riwayat_final, axis=0)

    prediksi_scaled = model.predict(input_tensor, verbose=0)
    
    prediksi_raw = scaler_y.inverse_transform(prediksi_scaled)
    
    estimasi_kali_nabung = math.ceil(prediksi_raw[0][0])

    return {
        "prediksi_raw": float(prediksi_raw[0][0]),
        "estimasi_kali_nabung": int(estimasi_kali_nabung)
    }