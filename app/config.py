# 환경변수 설정 — .env 파일에서 API 키 등 민감한 값을 읽어옴
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_next_question_model: str = "gpt-4o-mini"
    openai_turn_feedback_model: str = "gpt-5.4-mini"
    openai_session_feedback_model: str = "gpt-4o-mini"
    openai_fallback_model: str = "gpt-4o-mini"
    llm_provider: str = "openai"
    upstage_api_key: str | None = None
    upstage_base_url: str = "https://api.upstage.ai/v1"
    upstage_model: str = "solar-pro3"
    log_level: str = "INFO"
    sentry_dsn: str | None = None
    sentry_environment: str = "local"
    sentry_traces_sample_rate: float = 0.0
    sentry_max_breadcrumbs: int = 100

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
