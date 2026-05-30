# Sentry 초기화와 운영 오류 캡처를 담당한다.
import logging
from typing import Any

from app.config import settings
from app.core.logger import get_logger


logger = get_logger("observability")
_sentry_initialized = False
SENTRY_BREADCRUMB_LOG_LEVEL = logging.INFO
SENTRY_EVENT_LOG_LEVEL = None


def init_sentry(
    config=settings,
    *,
    sentry_sdk_module: Any | None = None,
    fastapi_integration_cls: type | None = None,
    logging_integration_cls: type | None = None,
) -> bool:
    dsn = _non_blank_string(getattr(config, "sentry_dsn", None))
    if dsn is None:
        logger.info("Sentry 비활성화 | reason=missing_dsn")
        return False

    sentry_sdk = sentry_sdk_module or _import_sentry_sdk()
    if sentry_sdk is None:
        logger.warning("Sentry SDK를 찾을 수 없어 초기화하지 않음 | environment=%s", config.sentry_environment)
        return False

    if fastapi_integration_cls is None or logging_integration_cls is None:
        imported_fastapi_cls, imported_logging_cls = _import_sentry_integrations()
        fastapi_integration_cls = fastapi_integration_cls or imported_fastapi_cls
        logging_integration_cls = logging_integration_cls or imported_logging_cls

    integrations = _build_sentry_integrations(fastapi_integration_cls, logging_integration_cls)

    sentry_sdk.init(
        dsn=dsn,
        environment=config.sentry_environment,
        traces_sample_rate=config.sentry_traces_sample_rate,
        max_breadcrumbs=config.sentry_max_breadcrumbs,
        integrations=integrations,
        send_default_pii=False,
    )
    global _sentry_initialized
    _sentry_initialized = True
    logger.info(
        "Sentry 초기화 완료 | environment=%s traces_sample_rate=%.2f",
        config.sentry_environment,
        config.sentry_traces_sample_rate,
    )
    return True


def _build_sentry_integrations(
    fastapi_integration_cls: type | None,
    logging_integration_cls: type | None,
) -> list[Any]:
    integrations = []
    if fastapi_integration_cls is not None:
        integrations.append(fastapi_integration_cls())
    if logging_integration_cls is not None:
        integrations.append(logging_integration_cls(
            level=SENTRY_BREADCRUMB_LOG_LEVEL,
            event_level=SENTRY_EVENT_LOG_LEVEL,
        ))
    return integrations


def capture_exception(exc: BaseException, *, sentry_sdk_module: Any | None = None) -> bool:
    sentry_sdk = sentry_sdk_module or _import_sentry_sdk()
    if sentry_sdk is None:
        return False

    sentry_sdk.capture_exception(exc)
    return True


def _non_blank_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _import_sentry_sdk():
    try:
        import sentry_sdk
    except ImportError:
        return None
    return sentry_sdk


def _import_sentry_integrations() -> tuple[type | None, type | None]:
    try:
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration
    except ImportError:
        return None, None
    return FastApiIntegration, LoggingIntegration
