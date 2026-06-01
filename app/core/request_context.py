# 요청 단위 추적 ID를 컨텍스트 변수로 보관한다.
from contextvars import ContextVar, Token


_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def get_request_id() -> str | None:
    return _request_id.get()


def set_request_id(request_id: str) -> Token:
    return _request_id.set(request_id)


def reset_request_id(token: Token) -> None:
    _request_id.reset(token)
