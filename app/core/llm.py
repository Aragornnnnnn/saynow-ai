# LLM 클라이언트 — OpenAI 호환 채팅 모델 호출의 단일 진입점
from dataclasses import dataclass

import openai
from app.config import settings
from app.core.logger import get_logger


@dataclass(frozen=True)
class LlmOptions:
    api_key: str
    model: str
    base_url: str | None = None


def resolve_llm_options(config=settings) -> LlmOptions:
    provider = config.llm_provider.lower()
    if provider == "upstage":
        if not config.upstage_api_key:
            raise RuntimeError("UPSTAGE_API_KEY is required when LLM_PROVIDER=upstage")
        return LlmOptions(
            api_key=config.upstage_api_key,
            model=config.upstage_model,
            base_url=config.upstage_base_url,
        )

    if provider == "openai":
        if not config.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        return LlmOptions(
            api_key=config.openai_api_key,
            model=config.openai_model,
        )

    raise RuntimeError(f"Unsupported LLM_PROVIDER: {config.llm_provider}")


OPTIONS = resolve_llm_options()
client = openai.OpenAI(
    api_key=OPTIONS.api_key,
    base_url=OPTIONS.base_url,
)
MODEL = OPTIONS.model
logger = get_logger("llm")


def chat(
    system: str,
    user: str,
    max_tokens: int = 1024,
    temperature: float = 0,
    timeout: float | None = None,
) -> str:
    logger.debug("LLM 호출 | user_prompt_preview: %s", user[:100].replace("\n", " "))
    request_options = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if timeout is not None:
        request_options["timeout"] = timeout
    response = client.chat.completions.create(**request_options)
    result = response.choices[0].message.content
    logger.debug("LLM 응답 | preview: %s", result[:100].replace("\n", " "))
    return result
