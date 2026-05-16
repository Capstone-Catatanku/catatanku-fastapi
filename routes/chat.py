from fastapi import APIRouter, HTTPException
from models.schemas import ChatRequest, ChatResponse
from services.llm_service import get_chat_response, clear_session

router = APIRouter(prefix="/api/chat", tags=["Chat"])

@router.post("/chatbot", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        reply = await get_chat_response(request.session_id, request.message)
        
        return ChatResponse(
            session_id=request.session_id,
            reply=reply
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error dari OpenRouter: {str(e)}")

@router.delete("/{session_id}")
async def reset_chat_endpoint(session_id: str):
    success = clear_session(session_id)
    if success:
        return {"message": f"Sesi {session_id} berhasil direset."}
    raise HTTPException(status_code=404, detail="Sesi tidak ditemukan.")