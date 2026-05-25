# 도움 요청 RAG 지식의 pgvector 검색과 저장을 담당한다.
from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.config import settings
from app.core.embeddings import embed_text
from app.core.logger import get_logger


logger = get_logger("assistance_rag")
UNSUPPORTED_DATABASE_QUERY_PARAMS = {
    "preparethreshold",
    "rewritebatchedinserts",
}


class NullAssistanceKnowledgeStore:
    def find_reusable_answer(self, request: Any) -> str | None:
        return None

    def save_interaction(self, request: Any, response: Any, *, answer_source: str) -> None:
        return None


class PgvectorAssistanceKnowledgeStore:
    def __init__(self, config=settings):
        self.config = config
        self.database_url = _normalize_database_url(
            config.assistance_rag_database_url or config.db_url or ""
        )
        self.database_username = config.assistance_rag_database_username or config.db_username
        self.database_password = config.assistance_rag_database_password or config.db_password
        self.table_name = config.assistance_rag_table

    def find_reusable_answer(self, request: Any) -> str | None:
        if not self.database_url:
            return None

        try:
            embedding = embed_text(_embedding_text_for_request(request), self.config)
            vector_literal = _vector_literal(embedding)
            table_identifier = _table_identifier(self.table_name)
            query = _sql().SQL(
                """
                select assistant_answer
                from {table}
                where embedding is not null
                  and scenario_title = %s
                  and quality_status in ('candidate', 'approved')
                  and 1 - (embedding <=> %s::vector) >= %s
                order by embedding <=> %s::vector
                limit %s
                """
            ).format(table=table_identifier)

            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        query,
                        (
                            request.scenarioTitle,
                            vector_literal,
                            self.config.assistance_rag_match_threshold,
                            vector_literal,
                            self.config.assistance_rag_match_count,
                        ),
                    )
                    row = cursor.fetchone()

            if not row:
                return None
            return row[0]
        except Exception as exc:
            logger.warning("도움 요청 RAG 검색 실패 | error: %s", exc)
            return None

    def save_interaction(self, request: Any, response: Any, *, answer_source: str) -> None:
        if not self.database_url or response.nextQuestion is None:
            return None

        try:
            embedding = embed_text(_embedding_text_for_request(request), self.config)
            vector_literal = _vector_literal(embedding)
            table_identifier = _table_identifier(self.table_name)
            quality_status = "candidate" if answer_source == "retrieved" else "generated"
            query = _sql().SQL(
                """
                insert into {table} (
                    scenario_category,
                    scenario_title,
                    scenario_goal,
                    original_question,
                    user_utterance,
                    assistant_answer,
                    turn_classification,
                    answer_source,
                    quality_status,
                    usage_count,
                    embedding
                )
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, 1, %s::vector)
                """
            ).format(table=table_identifier)

            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        query,
                        (
                            None,
                            request.scenarioTitle,
                            request.scenarioGoal,
                            request.originalQuestion,
                            request.userUtterance,
                            response.nextQuestion,
                            response.turnClassification.value,
                            answer_source,
                            quality_status,
                            vector_literal,
                        ),
                    )
                    if answer_source == "generated":
                        self._promote_repeated_generated_questions(cursor, table_identifier, request)
                connection.commit()
        except Exception as exc:
            logger.warning("도움 요청 RAG 저장 실패 | error: %s", exc)

    def _promote_repeated_generated_questions(self, cursor: Any, table_identifier: Any, request: Any) -> None:
        normalized_user_utterance = _normalize_repeated_question_key(request.userUtterance)
        repeat_threshold = self.config.assistance_rag_candidate_repeat_threshold
        query = _sql().SQL(
            """
            update {table}
            set quality_status = 'candidate',
                updated_at = now()
            where scenario_title = %s
              and trim(lower(regexp_replace(regexp_replace(user_utterance, '[^A-Za-z0-9[:space:]]', ' ', 'g'), '\\s+', ' ', 'g'))) = %s
              and quality_status = 'generated'
              and (
                  select count(*)
                  from {table}
                  where scenario_title = %s
                    and trim(lower(regexp_replace(regexp_replace(user_utterance, '[^A-Za-z0-9[:space:]]', ' ', 'g'), '\\s+', ' ', 'g'))) = %s
                    and quality_status = 'generated'
              ) >= %s
            """
        ).format(table=table_identifier)

        cursor.execute(
            query,
            (
                request.scenarioTitle,
                normalized_user_utterance,
                request.scenarioTitle,
                normalized_user_utterance,
                repeat_threshold,
            ),
        )

    def _connect(self):
        connection_kwargs = {}
        if self.database_username:
            connection_kwargs["user"] = self.database_username
        if self.database_password:
            connection_kwargs["password"] = self.database_password
        return _psycopg().connect(self.database_url, **connection_kwargs)


def build_assistance_knowledge_store(config=settings):
    if not config.assistance_rag_enabled:
        return NullAssistanceKnowledgeStore()

    database_url = config.assistance_rag_database_url or config.db_url
    if not database_url:
        return NullAssistanceKnowledgeStore()

    return PgvectorAssistanceKnowledgeStore(config)


def _embedding_text_for_request(request: Any) -> str:
    return "\n".join([
        f"Scenario title: {request.scenarioTitle}",
        f"Scenario situation: {request.scenarioSituation}",
        f"AI role: {request.aiRole}",
        f"Scenario goal: {request.scenarioGoal}",
        f"Previous AI question: {request.originalQuestion}",
        f"User utterance: {request.userUtterance}",
    ])


def _normalize_repeated_question_key(value: str) -> str:
    lowered = value.lower().strip()
    no_punctuation = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", no_punctuation).strip()


def _normalize_database_url(value: str) -> str:
    if value.startswith("jdbc:postgresql://"):
        value = "postgresql://" + value.removeprefix("jdbc:postgresql://")

    parsed = urlsplit(value)
    if not parsed.query:
        return value

    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    supported_params = [
        (key, query_value)
        for key, query_value in query_pairs
        if key.lower() not in UNSUPPORTED_DATABASE_QUERY_PARAMS
    ]
    if len(supported_params) == len(query_pairs):
        return value

    return urlunsplit((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        urlencode(supported_params),
        parsed.fragment,
    ))


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def _table_identifier(table_name: str):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?", table_name):
        raise ValueError("assistance_rag_table must be a table name or schema-qualified table name")
    return _sql().SQL(".").join(_sql().Identifier(part) for part in table_name.split("."))


def _psycopg():
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("psycopg is required when assistance RAG database is configured") from exc
    return psycopg


def _sql():
    try:
        from psycopg import sql
    except ImportError as exc:
        raise RuntimeError("psycopg is required when assistance RAG database is configured") from exc
    return sql
