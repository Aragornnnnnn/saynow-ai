-- 도움 요청 RAG 지식을 저장하는 pgvector 테이블을 생성한다.
create schema if not exists ai_rag;

create extension if not exists vector with schema extensions;

set search_path to public, extensions;

create table if not exists ai_rag.assistance_knowledge (
    id bigserial primary key,
    scenario_category text,
    scenario_title text not null,
    scenario_goal text not null,
    original_question text not null,
    user_utterance text not null,
    assistant_answer text not null,
    turn_classification text not null check (
        turn_classification in ('ANSWER', 'ASSISTANCE_REQUEST', 'INVALID_RESPONSE')
    ),
    answer_source text not null check (
        answer_source in ('generated', 'retrieved')
    ),
    quality_status text not null default 'generated' check (
        quality_status in ('generated', 'candidate', 'approved', 'rejected')
    ),
    usage_count integer not null default 1 check (usage_count >= 1),
    embedding vector(1536),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists assistance_knowledge_embedding_hnsw_idx
    on ai_rag.assistance_knowledge
    using hnsw (embedding vector_cosine_ops);

create index if not exists assistance_knowledge_lookup_idx
    on ai_rag.assistance_knowledge (scenario_title, quality_status);
