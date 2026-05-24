# 도움 요청 RAG 저장소의 설정 경계를 검증하는 테스트
import os
import unittest


os.environ.setdefault("OPENAI_API_KEY", "test-key")


class AssistanceKnowledgeStoreTest(unittest.TestCase):

    def test_build_store_returns_null_store_when_disabled(self):
        from app.config import Settings
        from app.services.assistance_knowledge_store import (
            NullAssistanceKnowledgeStore,
            build_assistance_knowledge_store,
        )

        settings = Settings(
            openai_api_key="test-key",
            db_url="postgresql://example",
            assistance_rag_enabled=False,
        )

        store = build_assistance_knowledge_store(settings)

        self.assertIsInstance(store, NullAssistanceKnowledgeStore)

    def test_build_store_returns_null_store_without_database_url(self):
        from app.config import Settings
        from app.services.assistance_knowledge_store import (
            NullAssistanceKnowledgeStore,
            build_assistance_knowledge_store,
        )

        settings = Settings(
            openai_api_key="test-key",
            db_url=None,
            assistance_rag_database_url=None,
        )

        store = build_assistance_knowledge_store(settings)

        self.assertIsInstance(store, NullAssistanceKnowledgeStore)

    def test_pgvector_store_normalizes_jdbc_database_url(self):
        from app.config import Settings
        from app.services.assistance_knowledge_store import PgvectorAssistanceKnowledgeStore

        settings = Settings(
            openai_api_key="test-key",
            assistance_rag_database_url="jdbc:postgresql://localhost:5432/saynow",
        )

        store = PgvectorAssistanceKnowledgeStore(settings)

        self.assertEqual(store.database_url, "postgresql://localhost:5432/saynow")

    def test_pgvector_store_removes_jdbc_only_query_parameters(self):
        from app.config import Settings
        from app.services.assistance_knowledge_store import PgvectorAssistanceKnowledgeStore

        settings = Settings(
            openai_api_key="test-key",
            assistance_rag_database_url=(
                "postgresql://localhost:5432/saynow"
                "?sslmode=require&prepareThreshold=0&reWriteBatchedInserts=true"
            ),
        )

        store = PgvectorAssistanceKnowledgeStore(settings)

        self.assertEqual(store.database_url, "postgresql://localhost:5432/saynow?sslmode=require")

    def test_pgvector_store_uses_database_credentials_from_shared_db_config(self):
        from app.config import Settings
        from app.services.assistance_knowledge_store import PgvectorAssistanceKnowledgeStore

        settings = Settings(
            openai_api_key="test-key",
            db_url="postgresql://localhost:5432/saynow",
            db_username="postgres.test-ref",
            db_password="secret",
        )

        store = PgvectorAssistanceKnowledgeStore(settings)

        self.assertEqual(store.database_username, "postgres.test-ref")
        self.assertEqual(store.database_password, "secret")


if __name__ == "__main__":
    unittest.main()
