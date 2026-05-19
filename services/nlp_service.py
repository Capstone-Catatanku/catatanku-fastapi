import httpx

HF_API_URL = "https://grace031105-catatanku-klasifikasi.hf.space/api/predict/kategori"

async def predict_kategori(deskripsi_transaksi: str):
    payload = {"deskripsi_transaksi": deskripsi_transaksi}
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(HF_API_URL, json=payload)
        response.raise_for_status()
        return response.json()