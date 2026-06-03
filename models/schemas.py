from pydantic import BaseModel
from typing import List

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    
class RiwayatTransaksi(BaseModel):
    target_nominal: float
    nominal_nabung: float
    total_terkumpul: float
    jarak_hari_nabung: int

class TabunganRequest(BaseModel):
    riwayat: List[RiwayatTransaksi]

class TabunganResponse(BaseModel):
    estimasi_kali_nabung: int
    prediksi_raw: float

class KategoriRequest(BaseModel):
    deskripsi_transaksi: str

class HasilSatu(BaseModel):
    transaksi: str
    llm_cleansed: str
    kategori: str
    confidence: float
    semua_skor: dict

class KategoriResponse(BaseModel):
    hasil: list[HasilSatu]