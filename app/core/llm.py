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

    if provider == "openrouter":
        if not config.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter")
        return LlmOptions(
            api_key=config.openrouter_api_key,
            model=config.openrouter_model,
            base_url=config.openrouter_base_url,
        )

    if provider == "openai":
        if not config.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        return LlmOptions(
            api_key=config.openai_api_key,
            model=config.openai_model,
        )

    raise RuntimeError(f"Unsupported LLM_PROVIDER: {config.llm_provider}")


def model_for_workflow(workflow: str, config=settings) -> str:
    provider = config.llm_provider.lower()
    if provider != "openai":
        return resolve_llm_options(config).model
    if workflow == "next_question":
        return config.openai_next_question_model or config.openai_model
    if workflow == "closing_message":
        return config.openai_closing_message_model or config.openai_model
    if workflow == "turn_feedback":
        return config.openai_turn_feedback_model or config.openai_model
    if workflow == "session_feedback":
        return config.openai_session_feedback_model or config.openai_model
    return config.openai_model


def fallback_model_for_workflow(workflow: str, config=settings) -> str | None:
    if config.llm_provider.lower() != "openai":
        return None
    primary_model = model_for_workflow(workflow, config)
    fallback_model = config.openai_fallback_model or config.openai_model
    if not fallback_model or fallback_model == primary_model:
        return None
    return fallback_model


OPTIONS = resolve_llm_options()
client = openai.OpenAI(
    api_key=OPTIONS.api_key,
    base_url=OPTIONS.base_url,
    timeout=settings.llm_request_timeout_seconds,
)
MODEL = OPTIONS.model
logger = get_logger("llm")


def chat(
    system: str,
    user: str,
    max_tokens: int = 1024,
    temperature: float = 0,
    model: str | None = None,
) -> str:
    selected_model = model or MODEL
    logger.debug("LLM 호출 | user_prompt_preview: %s", user[:100].replace("\n", " "))
    response = client.chat.completions.create(
        model=selected_model,
        temperature=temperature,
        **_token_limit_kwargs(selected_model, max_tokens),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    result = response.choices[0].message.content
    logger.debug("LLM 응답 | preview: %s", result[:100].replace("\n", " "))
    return result


def _token_limit_kwargs(model: str, max_tokens: int) -> dict[str, int]:
    if model.startswith("gpt-5"):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}
