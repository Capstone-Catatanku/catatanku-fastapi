from fastapi import FastAPI
from routes.chat import router as chat_router

# Inisialisasi Aplikasi FastAPI
app = FastAPI(
    title="Modular OpenRouter Chatbot",
    description="API Chatbot menggunakan OpenRouter (Gemini) dengan struktur modular.",
    version="1.0.0"
)

# Daftarkan router dari folder routes/
app.include_router(chat_router)

@app.get("/")
async def root():
    return {"message": "Server berjalan! Kunjungi /docs untuk mencoba API."}