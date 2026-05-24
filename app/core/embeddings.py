# 도움 요청 검색에 사용할 OpenAI 임베딩 호출을 감싼다.
import openai

from app.config import settings
from app.core.logger import get_logger


logger = get_logger("embeddings")


def embed_text(text: str, config=settings) -> list[float]:
    if not config.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required to generate embeddings")

    client = openai.OpenAI(api_key=config.openai_api_key)
    response = client.embeddings.create(
        model=config.openai_embedding_model,
        input=text,
    )
    embedding = response.data[0].embedding
    logger.debug("임베딩 생성 | model: %s | dimensions: %s", config.openai_embedding_model, len(embedding))
    return list(embedding)
