from fastapi import APIRouter, HTTPException
from models.schemas import KategoriRequest, KategoriResponse
from services.nlp_service import predict_kategori_lokal

router = APIRouter()

@router.post("/predict/kategori", response_model=KategoriResponse)
async def kategori(request: KategoriRequest):
    try:
        data_prediksi = await predict_kategori_lokal(request.deskripsi_transaksi)
        return KategoriResponse(hasil=data_prediksi)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))