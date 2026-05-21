import os
import json
import pickle
import re
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

SLANG_DICT = {
    "bli": "beli", "dapet": "dapat", "tdk": "tidak",
    "byr": "bayar", "gw": "saya", "dr": "dari", "yg": "yang",
    "dgn": "dengan", "utk": "untuk", "jd": "jadi", "krn": "karena",
    "udh": "sudah", "sdh": "sudah", "msh": "masih", "bs": "bisa",
    "nyicil": "cicilan", "cicil": "cicilan", "pesen": "pesan",
    "topup": "isi saldo", "kontrakan": "sewa",
    "invest": "investasi", "nyaham": "saham",
}

def preprocessing(teks: str) -> str:
    teks = teks.lower()
    teks = re.sub(r'[^\w\s]', '', teks)
    teks = re.sub(r'\s+', ' ', teks)
    return teks.strip()

def normalize_slang_words(teks: str) -> str:
    return ' '.join([SLANG_DICT.get(w, w) for w in teks.split()])

def clean(teks: str) -> str:
    teks = preprocessing(teks)
    teks = normalize_slang_words(teks)
    return teks

def split_transaksi(teks: str) -> list:
    parts = re.split(r',|\bdan\b|\blalu\b', teks)
    return [p.strip() for p in parts if p.strip()]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_MODELS_DIR = os.path.join(BASE_DIR, 'ml_models')

vocab_path = os.path.join(ML_MODELS_DIR, 'vocabulary.json')
with open(vocab_path, 'r', encoding='utf-8') as f:
    vocab = json.load(f)

vectorize_layer = TextVectorization(max_tokens=5000, output_sequence_length=20)
vectorize_layer.adapt(['dummy'])
vectorize_layer.set_vocabulary(vocab)

model_path = os.path.join(ML_MODELS_DIR, 'model_klasifikasi.keras')
model = tf.keras.models.load_model(model_path)

le_path = os.path.join(ML_MODELS_DIR, 'label_encoder.pkl')
with open(le_path, 'rb') as f:
    le = pickle.load(f)

async def predict_kategori_lokal(deskripsi_transaksi: str) -> list:
    transaksi_list = split_transaksi(deskripsi_transaksi)
    hasil = []
    
    for t in transaksi_list:
        teks_bersih = clean(t)
        
        pred = model.predict(tf.constant([teks_bersih]), verbose=0)
        proba = pred[0]
        idx = proba.argmax()
        
        hasil.append({
            "transaksi": t,
            "kategori": str(le.classes_[idx]),
            "confidence": round(float(proba[idx]) * 100, 2),
            "semua_skor": {
                str(le.classes_[i]): round(float(proba[i]) * 100, 2)
                for i in range(len(le.classes_))
            }
        })
        
    return hasil