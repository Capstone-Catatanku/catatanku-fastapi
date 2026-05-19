from pydantic import BaseModel

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    
class TabunganRequest(BaseModel):
    terkumpul: float
    target: float
    nabung: float

class TabunganResponse(BaseModel):
    estimasi_kali_nabung: int
    prediksi_raw: float
