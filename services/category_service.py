import os
import json
import pickle
import re
from google import genai
from google.genai import types
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import TextVectorization
from dotenv import load_dotenv

load_dotenv()

RAW_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY = RAW_API_KEY.strip().replace('"', '').replace("'", "") if RAW_API_KEY else None

client_sync = genai.Client(api_key=GEMINI_API_KEY)


def llm_clean(teks: str) -> str:
    if not GEMINI_API_KEY:
        print("[LLM] Error: GEMINI_API_KEY tidak ditemukan, pakai teks asli")
        return teks

    system_instruction_cleaner = """Kamu adalah normalisasi teks untuk aplikasi pencatatan keuangan Indonesia.
Tugasmu: ubah deskripsi transaksi menjadi frasa Bahasa Indonesia yang sederhana dan umum.
Aturan:
- Hapus nama orang dan nama tempat spesifik
- Ganti nama brand/aplikasi asing dengan kategori umumnya dalam Bahasa Indonesia
- Pertahankan makna aslinya, jangan ubah konteks transaksi
- Hasil harus 2-5 kata saja, Bahasa Indonesia
- Jangan tambahkan penjelasan, langsung tulis hasilnya saja
- Jangan menambahkan kata "tagihan" untuk sewa/kosan/kontrakan

Contoh:
- "makan ayam bakar di warung pak balil" → "makan ayam bakar"
- "beli netflix bulanan" → "langganan hiburan streaming"
- "top up gopay" → "isi saldo dompet digital"
- "bensin di SPBU Pertamina Batam" → "beli bahan bakar"
- "nyicil kosan cibaduyut" -> "cicilan kosan"
- "beli obat di klinik" → "beli obat kesehatan"
- "bayar kost bu Endah" → "bayar tempat tinggal" """

    try:
        response = client_sync.models.generate_content(
            model='gemini-2.5-flash',
            contents=teks,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction_cleaner,
                temperature=0.1,
                max_output_tokens=50,
            ),
        )
        return response.text.strip()
    except Exception as e:
        print(f"[LLM] Error SDK Gemini: {e}, pakai teks asli")
        return teks

@tf.keras.utils.register_keras_serializable()
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="normal"
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x, mask=None):
        e = tf.matmul(x, self.W)

        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            mask = tf.expand_dims(mask, -1)
            e = e + (mask - 1) * 1e9

        a = tf.nn.softmax(e, axis=1)
        output = x * a
        return tf.reduce_sum(output, axis=1)

    def compute_mask(self, inputs, mask=None):
        return None

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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in locals() else os.getcwd()
ML_MODELS_DIR = os.path.join(BASE_DIR, 'ml_models')

vocab_path = os.path.join(ML_MODELS_DIR, 'vocabulary.json')
with open(vocab_path, 'r', encoding='utf-8') as f:
    vocab = json.load(f)

vectorize_layer = TextVectorization(max_tokens=5000, output_sequence_length=30)
vectorize_layer.adapt(['dummy'])
vectorize_layer.set_vocabulary(vocab)

model_path = os.path.join(ML_MODELS_DIR, 'model_fitur_klasifikasi.keras')
model = tf.keras.models.load_model(
    model_path,
    custom_objects={"AttentionLayer": AttentionLayer}
)

le_path = os.path.join(ML_MODELS_DIR, 'label_encoder.pkl')
with open(le_path, 'rb') as f:
    le = pickle.load(f)

async def predict_kategori_lokal(deskripsi_transaksi: str) -> list:
    transaksi_list = split_transaksi(deskripsi_transaksi)
    hasil = []
    
    for t in transaksi_list:
        teks_llm = llm_clean(t)
        teks_bersih = clean(teks_llm)
        
        pred = model.predict(tf.constant([teks_bersih]), verbose=0)
        proba = pred[0]
        idx = proba.argmax()
        
        hasil.append({
            "transaksi": t,
            "llm_cleansed": teks_llm,
            "kategori": str(le.classes_[idx]),
            "confidence": round(float(proba[idx]) * 100, 2),
            "semua_skor": {
                str(le.classes_[i]): round(float(proba[i]) * 100, 2)
                for i in range(len(le.classes_))
            }
        })
        
    return hasil