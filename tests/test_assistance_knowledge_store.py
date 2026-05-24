# 도움 요청 RAG 저장소의 설정 경계를 검증하는 테스트
import os
import unittest
from types import SimpleNamespace


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

    def test_pgvector_store_promotes_repeated_generated_question_to_candidate(self):
        from app.config import Settings
        from app.models.conversation import NextQuestionTurnClassification
        from app.services import assistance_knowledge_store as store_module
        from app.services.assistance_knowledge_store import PgvectorAssistanceKnowledgeStore

        execute_calls = []

        class FakeCursor:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def execute(self, query, params=None):
                execute_calls.append((query, params))

        class FakeConnection:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def cursor(self):
                return FakeCursor()

            def commit(self):
                return None

        fake_psycopg = SimpleNamespace(
            connect=lambda *args, **kwargs: FakeConnection()
        )
        original_embed_text = store_module.embed_text
        original_psycopg = store_module._psycopg
        store_module.embed_text = lambda text, config: [0.1, 0.2, 0.3]
        store_module._psycopg = lambda: fake_psycopg

        try:
            settings = Settings(
                openai_api_key="test-key",
                db_url="postgresql://localhost:5432/saynow",
                assistance_rag_candidate_repeat_threshold=2,
            )
            store = PgvectorAssistanceKnowledgeStore(settings)
            request = SimpleNamespace(
                scenarioTitle="카페에서 주문하기",
                scenarioGoal="원하는 음료를 자연스럽게 주문할 수 있다.",
                originalQuestion="What would you like to order?",
                userUtterance="  Can   I see the MENU?  ",
            )
            response = SimpleNamespace(
                nextQuestion="We have coffee, tea, and smoothies. What would you like to order?",
                turnClassification=NextQuestionTurnClassification.ASSISTANCE_REQUEST,
            )

            store.save_interaction(request, response, answer_source="generated")
        finally:
            store_module.embed_text = original_embed_text
            store_module._psycopg = original_psycopg

        self.assertEqual(len(execute_calls), 2)
        self.assertEqual(
            execute_calls[1][1],
            ("카페에서 주문하기", "can i see the menu", "카페에서 주문하기", "can i see the menu", 2),
        )

    def test_repeated_question_key_ignores_case_spacing_and_punctuation(self):
        from app.services.assistance_knowledge_store import _normalize_repeated_question_key

        self.assertEqual(
            _normalize_repeated_question_key("  Can   I see the MENU?  "),
            "can i see the menu",
        )
        self.assertEqual(
            _normalize_repeated_question_key("That's all."),
            "that s all",
        )


if __name__ == "__main__":
    unittest.main()
