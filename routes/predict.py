from fastapi import APIRouter
from models.schemas import TabunganRequest, TabunganResponse
from services.ml_service import predict_estimasi_tabungan

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)

@router.post("/tabungan", response_model=TabunganResponse)
async def hitung_tabungan(data: TabunganRequest):
    hasil = predict_estimasi_tabungan(data.terkumpul, data.target, data.nabung)
    
    return TabunganResponse(**hasil)