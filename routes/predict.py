from fastapi import APIRouter, HTTPException
from models.schemas import TabunganRequest, TabunganResponse
from services.ml_service import predict_estimasi_tabungan

router = APIRouter(
    prefix="/api/predict",
    tags=["Prediction"]
)

@router.post("/tabungan", response_model=TabunganResponse)
async def hitung_tabungan(data: TabunganRequest):
    try:
        hasil = predict_estimasi_tabungan(data.terkumpul, data.target, data.nabung)
        return TabunganResponse(**hasil)
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Terjadi kesalahan pada server saat menghitung tabungan: {str(e)}"
        )