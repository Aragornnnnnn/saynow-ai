# 환경변수 설정 — .env 파일에서 API 키 등 민감한 값을 읽어옴
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
