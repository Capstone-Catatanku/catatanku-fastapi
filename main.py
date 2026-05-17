from fastapi import FastAPI
from routes.chat import router as chat_router
from routes.predict import router as predict_router

app = FastAPI(
    title="Catatanku",
    description="API Catatanku untuk memprediksi tabungan, mengklasifikasikan kategori dan chatbot",
    version="1.0.0"
)

app.include_router(chat_router)
app.include_router(predict_router)

@app.get("/")
async def root():
    return {"message": "Server berjalan! Kunjungi /docs untuk mencoba API."}