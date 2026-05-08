# TTS 라우터 — POST /tts (텍스트 → base64 MP3 오디오 반환)
from fastapi import APIRouter
from pydantic import BaseModel
from app.services.tts_service import synthesize

router = APIRouter(prefix="/tts", tags=["tts"])


class TTSRequest(BaseModel):
    text: str
    voice: str = "alloy"


@router.post("", summary="음성 합성 (TTS)")
def text_to_speech(body: TTSRequest):
    """
    텍스트를 OpenAI TTS로 자연스러운 음성으로 변환합니다.

    - text: 읽을 텍스트
    - voice: 목소리 선택 (alloy, echo, fable, onyx, nova, shimmer)
    - 반환: MP3 음성 파일 (base64 인코딩)
    """
    try:
        audio_base64 = synthesize(body.text, voice=body.voice)
        return {"success": True, "data": {"audio_base64": audio_base64}, "error": None}
    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}
