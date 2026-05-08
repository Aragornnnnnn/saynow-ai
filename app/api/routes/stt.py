# STT 라우터 — POST /stt (오디오 파일 업로드 → 영어 텍스트 반환)
from fastapi import APIRouter, UploadFile, File
from app.services.stt_service import transcribe

router = APIRouter(prefix="/stt", tags=["stt"])


@router.post("", summary="음성 인식 (STT)")
async def speech_to_text(audio: UploadFile = File(...)):
    """
    오디오 파일을 업로드하면 OpenAI Whisper로 영어 텍스트로 변환합니다.

    - audio: 음성 파일 (webm, mp3, wav 등)
    - 반환: 인식된 영어 텍스트
    """
    try:
        audio_bytes = await audio.read()
        text = transcribe(audio_bytes, filename=audio.filename or "audio.webm")
        return {"success": True, "data": {"text": text}, "error": None}
    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}
