# OpenAI 클라이언트 — LLM 호출의 단일 진입점, 모든 서비스가 이걸 통해 GPT를 씀
import openai
from app.config import settings

client = openai.OpenAI(api_key=settings.openai_api_key)
MODEL = "gpt-4o-mini"


def chat(system: str, user: str, max_tokens: int = 1024) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content
