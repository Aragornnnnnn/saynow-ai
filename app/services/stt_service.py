# STT 서비스 — 오디오 바이트를 OpenAI Whisper API로 전송해 영어 텍스트 및 신뢰도로 변환
import openai
from app.config import settings

_client = openai.OpenAI(api_key=settings.openai_api_key)


def transcribe(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    """기존 호환용 — 텍스트만 반환"""
    result = transcribe_with_confidence(audio_bytes, filename)
    return result["text"]


def transcribe_with_confidence(audio_bytes: bytes, filename: str = "audio.webm") -> dict:
    """텍스트와 평균 신뢰도를 함께 반환 {"text": str, "confidence": float}"""
    response = _client.audio.transcriptions.create(
        model="whisper-1",
        file=(filename, audio_bytes),
        language="en",
        response_format="verbose_json",
    )
    text = response.text

    # segment별 avg_logprob → confidence (0~1) 평균
    import math
    segments = getattr(response, "segments", None) or []
    if segments:
        logprobs = [
            segment.get("avg_logprob") if isinstance(segment, dict) else getattr(segment, "avg_logprob", None)
            for segment in segments
        ]
        logprobs = [logprob for logprob in logprobs if logprob is not None]
        if not logprobs:
            return {"text": text, "confidence": 0.5}
        avg_logprob = sum(logprobs) / len(logprobs)
        confidence = round(min(1.0, max(0.0, math.exp(avg_logprob))), 4)
    else:
        confidence = 0.5  # segment 없으면 중립값

    return {"text": text, "confidence": confidence}
