# FastAPI 앱 진입점 — 라우터 등록, CORS 설정, 서버 실행 시작점
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import scenario, stt, tts
from app.api.routes import turn_evaluation, session_feedback

app = FastAPI(title="SayNow API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(scenario.router)
app.include_router(stt.router)
app.include_router(tts.router)
app.include_router(turn_evaluation.router)
app.include_router(session_feedback.router)


@app.get("/health")
def health():
    return {"status": "ok"}
