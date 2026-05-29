import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

RAW_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY = RAW_API_KEY.strip().replace('"', '').replace("'", "") if RAW_API_KEY else None

client = genai.Client(api_key=GEMINI_API_KEY)

chat_sessions = {}

SYSTEM_PROMPT = """
Kamu adalah seorang Penasihat Keuangan Profesional.
Aturan ketat yang HARUS kamu patuhi:
1. Kamu HANYA boleh menjawab pertanyaan yang berkaitan dengan keuangan, investasi, ekonomi, akuntansi, dan pengelolaan uang.
2. Jika pengguna bertanya tentang topik di luar keuangan (misalnya: pemrograman, sejarah, cuaca, kesehatan, atau hal umum lainnya), kamu harus menolak menjawabnya dengan sopan.
3. Saat menolak, ingatkan pengguna bahwa keahlianmu hanya di bidang keuangan.
4. Gunakan bahasa yang profesional namun mudah dipahami.
"""

async def get_chat_response(session_id: str, user_message: str) -> str:
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY belum dikonfigurasi di file .env"

    if session_id not in chat_sessions:
        chat_sessions[session_id] = client.aio.chats.create(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2
            )
        )
    
    try:
        response = await chat_sessions[session_id].send_message(user_message)
        return response.text

    except Exception as e:
        print(f"[ChatBot] Error SDK Gemini: {e}")
        return f"Maaf, terjadi kesalahan internal pada sistem bot: {e}"

def clear_session(session_id: str) -> bool:
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return True
    return False