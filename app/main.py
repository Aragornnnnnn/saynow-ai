# FastAPI 앱 진입점 — 2차 MVP 대화 API 라우터와 공통 예외 처리를 등록한다.
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import conversation
from app.core.logger import get_logger
from app.core.observability import capture_exception, init_sentry

app = FastAPI(title="SayNow AI API", version="0.2.0")
logger = get_logger("main")
init_sentry()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error("[400] 요청 DTO 검증 실패 — 필드 타입 오류 또는 필수 필드 누락 (클라이언트 문제) | path: %s | error: %s", request.url.path, exc.errors())
    return JSONResponse(
        status_code=400,
        content={
            "code": "INVALID_REQUEST",
            "message": "잘못된 요청입니다.",
        },
    )


@app.exception_handler(Exception)
async def internal_exception_handler(request: Request, exc: Exception):
    logger.error(
        "[500] 서버 내부 오류 — 예상치 못한 예외 발생 | path: %s | error: %s",
        request.url.path,
        exc,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    capture_exception(exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(conversation.router)


@app.get("/health")
def health():
    return {"status": "ok"}
