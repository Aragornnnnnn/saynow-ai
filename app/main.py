# FastAPI 앱 진입점 — 라우터 등록, CORS 설정, 서버 실행 시작점
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import scenario, stt, tts
from app.api.routes import turn_evaluation, session_feedback
from app.core.logger import get_logger

app = FastAPI(title="SayNow API", version="0.1.0")
logger = get_logger("main")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error("[422] 요청 DTO 검증 실패 — 필드 타입 오류 또는 필수 필드 누락 (클라이언트 문제) | path: %s | error: %s", request.url.path, exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def internal_exception_handler(request: Request, exc: Exception):
    logger.error("[500] 서버 내부 오류 — 예상치 못한 예외 발생 | path: %s | error: %s", request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

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
