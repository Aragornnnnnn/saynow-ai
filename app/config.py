# 환경변수 설정 — .env 파일에서 API 키 등 민감한 값을 읽어옴
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    llm_provider: str = "openai"
    upstage_api_key: str | None = None
    upstage_base_url: str = "https://api.upstage.ai/v1"
    upstage_model: str = "solar-pro3"
    db_url: str | None = None
    db_username: str | None = None
    db_password: str | None = None
    assistance_rag_enabled: bool = True
    assistance_rag_database_url: str | None = None
    assistance_rag_database_username: str | None = None
    assistance_rag_database_password: str | None = None
    assistance_rag_table: str = "ai_rag.assistance_knowledge"
    assistance_rag_match_threshold: float = 0.78
    assistance_rag_match_count: int = 3
    assistance_rag_candidate_repeat_threshold: int = 2
    log_level: str = "INFO"
    sentry_dsn: str | None = None
    sentry_environment: str = "local"
    sentry_traces_sample_rate: float = 0.0

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
