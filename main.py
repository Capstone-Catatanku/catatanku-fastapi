import os
from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from routes.chat import router as chat_router
from routes.predict import router as predict_router

settings = {
    "title": "Catatanku",
    "description": "API Catatanku untuk memprediksi tabungan, mengklasifikasikan kategori dan chatbot",
    "version": "1.0.0",
    "docs_url": None,
    "redoc_url": None
}

if os.environ.get("SPACE_ID"):
    settings["root_path"] = f"/spaces/{os.environ.get('SPACE_ID')}"

app = FastAPI(**settings)

app.include_router(chat_router)
app.include_router(predict_router)

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    openapi_url = f"{app.root_path}{app.openapi_url}" if app.root_path else app.openapi_url
    
    return get_swagger_ui_html(
        openapi_url=openapi_url,
        title=app.title + " - Swagger UI",
        swagger_js_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css",
    )

@app.get("/")
async def root():
    return {"message": "Server berjalan! Kunjungi /docs untuk mencoba API."}