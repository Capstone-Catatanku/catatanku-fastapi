from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.nlp_service import predict_kategori

router = APIRouter()

class KategoriRequest(BaseModel):
    deskripsi_transaksi: str

@router.post("/predict/kategori")
async def kategori(request: KategoriRequest):
    try:
        result = await predict_kategori(request.deskripsi_transaksi)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))