# 📒 Catatanku API

Backend API untuk aplikasi pencatatan keuangan **Catatanku**, dibangun dengan **FastAPI**. API ini menyediakan tiga fitur utama berbasis Machine Learning dan AI: klasifikasi kategori transaksi, prediksi estimasi tabungan, dan chatbot penasihat keuangan.

---

## ✨ Fitur

| Fitur | Deskripsi |
|---|---|
| 🏷️ Klasifikasi Kategori | Mengklasifikasikan deskripsi transaksi ke kategori keuangan secara otomatis menggunakan model TensorFlow + normalisasi teks via Gemini AI |
| 💰 Prediksi Tabungan | Memprediksi estimasi berapa kali lagi nabung untuk mencapai target, menggunakan model LSTM |
| 🤖 Chatbot Keuangan | Chatbot berbasis Gemini 2.5 Flash yang hanya menjawab pertanyaan seputar keuangan, investasi, dan ekonomi |

---

## 🛠️ Tech Stack

- **Framework**: FastAPI + Uvicorn
- **ML/AI**: TensorFlow/Keras (LSTM, Attention Layer, TextVectorization)
- **LLM**: Google Gemini 2.5 Flash (via `google-genai`)
- **Data**: NumPy, Pandas, Scikit-learn, Joblib
- **Config**: Python-dotenv
- **Containerization**: Docker
- **CI/CD**: GitHub Actions → Hugging Face Spaces

---

## 📁 Struktur Proyek

```
catatanku-fastapi/
├── main.py                        # Entry point FastAPI
├── requirements.txt
├── Dockerfile
├── .env                           # Environment variable (tidak di-commit)
├── .env.example                   # Contoh konfigurasi environment variable
├── models/
│   └── schemas.py                 # Pydantic request/response models
├── routes/
│   ├── category.py                # Route klasifikasi kategori
│   ├── savings.py                 # Route prediksi tabungan
│   └── chatbot.py                 # Route chatbot
├── services/
│   ├── category_service.py        # Logic klasifikasi + LLM cleansing
│   ├── savings_service.py         # Logic prediksi LSTM tabungan
│   └── chatbot_service.py         # Logic chatbot Gemini
└── ml_models/                     # (tidak di-commit) Model & artefak ML
    ├── model_klasifikasi.keras
    ├── model_lstm_tabungan.keras
    ├── vocabulary.json
    ├── label_encoder.pkl
    ├── scaler_lstm.pkl
    └── scaler_y_lstm.pkl
```

---

## 🚀 Cara Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/<username>/catatanku-fastapi.git
cd catatanku-fastapi
```

### 2. Buat Virtual Environment & Install Dependency

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 3. Konfigurasi Environment Variable

Buat file `.env` di root proyek:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

> Dapatkan API key Gemini di [Google AI Studio](https://aistudio.google.com/app/apikey).

### 4. Siapkan File ML Models

File model ML dikelola di repository terpisah. Clone terlebih dahulu:

```bash
git clone https://github.com/Capstone-Catatanku/ai-model.git
```

Kemudian salin atau pindahkan hasilnya ke folder `ml_models/` di root proyek ini:

```
ml_models/
├── model_klasifikasi.keras
├── model_lstm_tabungan.keras
├── vocabulary.json
├── label_encoder.pkl
├── scaler_lstm.pkl
└── scaler_y_lstm.pkl
```

### 5. Jalankan Server

```bash
uvicorn main:app --reload
```

Server berjalan di `http://localhost:8000`. Buka `http://localhost:8000/docs` untuk Swagger UI.

---

## 🐳 Menjalankan dengan Docker

```bash
# Build image
docker build -t catatanku-api .

# Jalankan container
docker run -p 7860:7860 --env-file .env catatanku-api
```

Server berjalan di `http://localhost:7860`.

---

## 📡 Dokumentasi API

### 🏷️ Klasifikasi Kategori Transaksi

**`POST /api/predict/kategori`**

Mengklasifikasikan satu atau lebih transaksi (dipisahkan koma atau kata "dan"/"lalu") ke kategori keuangan.

**Request Body:**
```json
{
  "deskripsi_transaksi": "makan ayam bakar di warung, beli bensin, bayar kosan"
}
```

**Response:**
```json
{
  "hasil": [
    {
      "transaksi": "makan ayam bakar di warung",
      "llm_cleansed": "makan ayam bakar",
      "kategori": "Makanan & Minuman",
      "confidence": 97.43,
      "semua_skor": {
        "Makanan & Minuman": 97.43,
        "Transportasi": 0.82,
        "...": "..."
      }
    }
  ]
}
```

---

### 💰 Prediksi Estimasi Tabungan

**`POST /api/predict/tabungan`**

Memprediksi berapa kali lagi perlu menabung untuk mencapai target berdasarkan riwayat transaksi (minimal 1, optimal 5 data terakhir).

**Request Body:**
```json
{
  "riwayat": [
    {
      "target_nominal": 5000000,
      "nominal_nabung": 500000,
      "total_terkumpul": 1500000,
      "jarak_hari_nabung": 7
    }
  ]
}
```

**Response:**
```json
{
  "estimasi_kali_nabung": 8,
  "prediksi_raw": 7.62
}
```

---

### 🤖 Chatbot Penasihat Keuangan

**`POST /api/chat/ask`**

Mengirim pesan ke chatbot. Chatbot hanya menjawab pertanyaan seputar keuangan, investasi, ekonomi, dan akuntansi.

**Request Body:**
```json
{
  "session_id": "user-abc123",
  "message": "Bagaimana cara mulai investasi reksa dana?"
}
```

**Response:**
```json
{
  "session_id": "user-abc123",
  "reply": "Untuk mulai investasi reksa dana, langkah pertama adalah..."
}
```

---

**`DELETE /api/chat/{session_id}`**

Menghapus/mereset sesi percakapan chatbot.

**Response:**
```json
{
  "message": "Sesi user-abc123 berhasil direset."
}
```

---

## ⚙️ CI/CD — Deploy ke Hugging Face Spaces

Setiap push ke branch `main` akan otomatis melakukan sync ke [Hugging Face Spaces](https://huggingface.co/spaces) via GitHub Actions.

**Setup yang diperlukan:**

1. Buat secret `HF_TOKEN` di repository GitHub (Settings → Secrets → Actions).
2. Buat HF Space bertipe **Docker** dengan nama yang sesuai di `deploy.yml`.

---

## 🤲 Deploy Manual ke Hugging Face Spaces

Jika ingin deploy tanpa GitHub Actions, ikuti langkah berikut.

### 1. Install Git LFS

Hugging Face menggunakan Git LFS untuk file besar (model `.keras`, `.pkl`, dll.).

```bash
# Linux
sudo apt install git-lfs

# Mac
brew install git-lfs

# Windows — download installer dari https://git-lfs.com

git lfs install
```

### 2. Clone Repository HF Space

```bash
git clone https://huggingface.co/spaces/<username-hf>/<nama-space>
cd <nama-space>
```

> Ganti `<username-hf>` dan `<nama-space>` sesuai akun Hugging Face kamu.

### 3. Salin Semua File Proyek

Salin seluruh isi folder `catatanku-fastapi` ke dalam folder HF Space yang baru di-clone:

```bash
cp -r /path/to/catatanku-fastapi/* .
```

Pastikan folder `ml_models/` beserta semua file model juga ikut tersalin. File model bisa di-clone dari [repository model](https://github.com/Capstone-Catatanku/ai-model.git):

```
ml_models/
├── model_klasifikasi.keras
├── model_lstm_tabungan.keras
├── vocabulary.json
├── label_encoder.pkl
├── scaler_lstm.pkl
└── scaler_y_lstm.pkl
```

### 4. Tambahkan File `.env` atau Konfigurasi Secret

Jangan commit file `.env` ke HF. Sebagai gantinya, tambahkan secret langsung di dashboard:

1. Buka halaman Space di `https://huggingface.co/spaces/<username-hf>/<nama-space>`.
2. Klik **Settings** → **Variables and secrets**.
3. Tambahkan secret baru: `GEMINI_API_KEY` → isi dengan API key kamu.

### 5. Track File Besar dengan Git LFS

```bash
git lfs track "*.keras"
git lfs track "*.pkl"
git add .gitattributes
```

### 6. Commit dan Push

```bash
git add .
git commit -m "deploy: upload catatanku-fastapi"
git push
```

> Jika diminta autentikasi, gunakan username HF dan **token** (bukan password). Buat token di [Settings → Access Tokens](https://huggingface.co/settings/tokens) dengan role **Write**.

### 7. Pantau Build

Setelah push berhasil, buka halaman Space kamu. Tab **Logs** akan menampilkan proses build Docker secara live. Tunggu hingga status berubah menjadi **Running** 🟢.

---

## 📝 Catatan Pengembangan

- Model ML tidak disertakan di repository ini. Letakkan file model secara manual di folder `ml_models/` sebelum menjalankan server.
- `GEMINI_API_KEY` wajib dikonfigurasi; tanpanya fitur klasifikasi kategori dan chatbot tidak akan berfungsi.
- Sesi chatbot disimpan **in-memory**; sesi akan hilang ketika server di-restart.
