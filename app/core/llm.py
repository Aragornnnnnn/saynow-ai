# OpenAI 클라이언트 — LLM 호출의 단일 진입점, 모든 서비스가 이걸 통해 GPT를 씀
import openai
from app.config import settings
from app.core.logger import get_logger

client = openai.OpenAI(api_key=settings.openai_api_key)
MODEL = "gpt-4o-mini"
logger = get_logger("llm")


def chat(system: str, user: str, max_tokens: int = 1024) -> str:
    logger.debug("LLM 호출 | user_prompt_preview: %s", user[:100].replace("\n", " "))
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    result = response.choices[0].message.content
    logger.debug("LLM 응답 | preview: %s", result[:100].replace("\n", " "))
    return result
