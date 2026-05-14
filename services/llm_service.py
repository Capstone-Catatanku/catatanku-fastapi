import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

chat_sessions = {}
MODEL_NAME = "deepseek/deepseek-v4-flash:free"

# === TAMBAHKAN SYSTEM PROMPT DI SINI ===
SYSTEM_PROMPT = """
Kamu adalah seorang Penasihat Keuangan Profesional.
Aturan ketat yang HARUS kamu patuhi:
1. Kamu HANYA boleh menjawab pertanyaan yang berkaitan dengan keuangan, investasi, ekonomi, akuntansi, dan pengelolaan uang.
2. Jika pengguna bertanya tentang topik di luar keuangan (misalnya: pemrograman, sejarah, cuaca, kesehatan, atau hal umum lainnya), kamu harus menolak menjawabnya dengan sopan.
3. Saat menolak, ingatkan pengguna bahwa keahlianmu hanya di bidang keuangan.
4. Gunakan bahasa yang profesional namun mudah dipahami.
"""

async def get_chat_response(session_id: str, user_message: str) -> str:
    if session_id not in chat_sessions:
        chat_sessions[session_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
    
    chat_sessions[session_id].append({"role": "user", "content": user_message})
    
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=chat_sessions[session_id],
        extra_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "FastAPI Finance Bot",
        }
    )
    
    ai_reply = response.choices[0].message.content
    
    chat_sessions[session_id].append({"role": "assistant", "content": ai_reply})
    
    return ai_reply

def clear_session(session_id: str) -> bool:
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return True
    return False