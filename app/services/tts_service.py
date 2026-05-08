# TTS 서비스 — 텍스트를 OpenAI TTS API로 음성 합성해 base64 인코딩된 MP3로 반환
import base64
import openai
from app.config import settings

_client = openai.OpenAI(api_key=settings.openai_api_key)


def synthesize(text: str, voice: str = "alloy") -> str:
    """Returns base64-encoded MP3 audio."""
    response = _client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
    )
    return base64.b64encode(response.content).decode("utf-8")
