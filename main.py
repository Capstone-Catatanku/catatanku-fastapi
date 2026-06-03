from fastapi import FastAPI
from routes.chatbot import router as chatbot_router
from routes.savings import router as savings_router
from routes.category import router as category_router

app = FastAPI(
    title="Catatanku",
    description="API Catatanku untuk memprediksi tabungan, mengklasifikasikan kategori dan chatbot",
    version="1.0.0"
)

app.include_router(chatbot_router)
app.include_router(savings_router)
app.include_router(category_router, prefix="/api", tags=["category"])

@app.get("/")
async def root():
    return {"message": "Server berjalan! Kunjungi /docs untuk mencoba API."}