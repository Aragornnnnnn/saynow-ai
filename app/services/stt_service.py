# STT 서비스 — 오디오 바이트를 OpenAI Whisper API로 전송해 영어 텍스트로 변환
import openai
from app.config import settings

_client = openai.OpenAI(api_key=settings.openai_api_key)


def transcribe(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    response = _client.audio.transcriptions.create(
        model="whisper-1",
        file=(filename, audio_bytes),
        language="en",
    )
    return response.text
