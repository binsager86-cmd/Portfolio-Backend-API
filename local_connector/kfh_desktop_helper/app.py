"""Desktop helper HTTP boundary.

Reuses kfh_gate5b's KfhLocalTestController - and its Pydantic request
models - unmodified: nothing about *how this talks to KFH* changes. Only
the HTTP boundary around it is new, because the dev/test tool's boundary
(local_connector.kfh_gate5b.local_api.create_local_test_app) hard-requires
an explicit port in the Origin header (proven by
tests/unit/test_kfh_gate5b_l2_r3_origin.py::test_malformed_origins_are_format_invalid,
which asserts "http://localhost" with no port is rejected as malformed).
Real browsers omit the port for a site's default port (443 for https, 80
for http), so that dev-only boundary can never accept a production site's
Origin header. This module's canonicalize_browser_origin treats an omitted
port as the scheme default instead, which is what a real deployed site
needs and a fixed localhost:8081 dev port never had to handle.

Everything else - loopback-only binding, a per-request nonce, no request
logging, no credential persistence - mirrors the dev/test tool's posture.
"""

from __future__ import annotations

import secrets
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from local_connector.kfh_gate5b.local_api import (
    PREVIEW_HANDOFF_VERSION,
    KfhLocalTestController,
    LocalAccountSelectionRequest,
    LocalConnectAndFetchRequest,
    LocalOtpRequest,
    _model_validation_failure,
    _validation_failure,
)

NONCE_HEADER = "X-KFH-Local-Nonce"
LOCAL_HOST = "127.0.0.1"
LOCAL_PORT = 8765
HELPER_PATH_PREFIX = "/kfh-helper/"


def canonicalize_browser_origin(value: str) -> str:
    """Canonicalize a URL origin, treating an omitted port as the scheme's
    default (443 for https, 80 for http) - how real browsers send Origin
    for a site on its default port, unlike kfh_gate5b's stricter
    localhost-dev-only variant which requires the port to be explicit."""
    if not isinstance(value, str) or not value:
        raise ValueError("origin must be a non-empty string")
    parsed = urlsplit(value)
    scheme = parsed.scheme.lower()
    if (
        scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("origin must contain only scheme, hostname, and an optional port")
    default_port = 443 if scheme == "https" else 80
    port = parsed.port if parsed.port is not None else default_port
    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    return f"{scheme}://{host}:{port}"


def create_desktop_helper_app(
    *,
    controller: KfhLocalTestController | Any | None = None,
    nonce: str,
    allowed_origin: str,
) -> FastAPI:
    if len(nonce) < 32:
        raise ValueError("KFH desktop helper pairing nonce must contain at least 32 characters")
    canonical_allowed_origin = canonicalize_browser_origin(allowed_origin)
    local_controller = controller or KfhLocalTestController()

    app = FastAPI(
        title="Saham KFH Desktop Helper",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.allowed_origin = canonical_allowed_origin
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[canonical_allowed_origin],
        allow_credentials=False,
        allow_private_network=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", NONCE_HEADER],
    )

    @app.exception_handler(RequestValidationError)
    async def _sanitized_validation_failure(
        _request: Request, error: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content=_model_validation_failure(error.errors()))

    def _no_store(response: Any) -> Any:
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        return response

    @app.middleware("http")
    async def hardened_helper_boundary(request: Request, call_next: Any) -> Any:
        host = request.client.host if request.client else ""
        if host not in {"127.0.0.1", "::1"}:
            return _no_store(
                JSONResponse(
                    status_code=403,
                    content=_validation_failure("ORIGIN", "INVALID_FORMAT", failure_code="ORIGIN_REJECTED"),
                )
            )

        if request.method == "OPTIONS":
            return _no_store(await call_next(request))

        supplied_origin = request.headers.get("origin")
        if supplied_origin is None:
            return _no_store(
                JSONResponse(
                    status_code=403,
                    content=_validation_failure(
                        "ORIGIN", "MISSING", failure_code="ORIGIN_REJECTED", origin_status="REJECTED"
                    ),
                )
            )
        try:
            canonical_supplied_origin = canonicalize_browser_origin(supplied_origin)
        except ValueError:
            return _no_store(
                JSONResponse(
                    status_code=403,
                    content=_validation_failure(
                        "ORIGIN",
                        "FORMAT_INVALID",
                        failure_code="ORIGIN_REJECTED",
                        origin_status="REJECTED",
                    ),
                )
            )
        if canonical_supplied_origin != canonical_allowed_origin:
            return _no_store(
                JSONResponse(
                    status_code=403,
                    content=_validation_failure(
                        "ORIGIN",
                        "NOT_ALLOWED",
                        failure_code="ORIGIN_REJECTED",
                        origin_status="REJECTED",
                    ),
                )
            )

        if request.url.path.startswith(HELPER_PATH_PREFIX):
            supplied_nonce = request.headers.get(NONCE_HEADER)
            if supplied_nonce is None:
                return _no_store(
                    JSONResponse(
                        status_code=403,
                        content=_validation_failure(
                            "NONCE", "MISSING", failure_code="NONCE_REJECTED", nonce_status="REJECTED"
                        ),
                    )
                )
            if not secrets.compare_digest(supplied_nonce, nonce):
                return _no_store(
                    JSONResponse(
                        status_code=403,
                        content=_validation_failure(
                            "NONCE",
                            "INVALID_FORMAT",
                            failure_code="NONCE_REJECTED",
                            nonce_status="REJECTED",
                        ),
                    )
                )

        return _no_store(await call_next(request))

    @app.get(f"{HELPER_PATH_PREFIX}health")
    async def health() -> dict[str, Any]:
        # Same shape as kfh_gate5b's /health (the frontend's health() parser
        # requires an exact key set) - localTestEnabled is a legacy field
        # name from that shared contract, not a claim about this being the
        # dev/test tool.
        return {
            "status": "OK",
            "localTestEnabled": True,
            "previewHandoffVersion": PREVIEW_HANDOFF_VERSION,
            "serverAllowedOrigin": canonical_allowed_origin,
            "serverAllowedOriginConfigured": True,
        }

    @app.post(f"{HELPER_PATH_PREFIX}connect-and-fetch")
    async def connect(payload: LocalConnectAndFetchRequest) -> Any:
        return await local_controller.connect(payload)

    @app.get(f"{HELPER_PATH_PREFIX}status")
    async def status() -> dict[str, Any]:
        return await local_controller.status()

    @app.post(f"{HELPER_PATH_PREFIX}otp")
    async def submit_otp(payload: LocalOtpRequest) -> dict[str, Any]:
        return await local_controller.submit_otp(payload)

    @app.post(f"{HELPER_PATH_PREFIX}select-account")
    async def select_account(payload: LocalAccountSelectionRequest) -> dict[str, Any]:
        return await local_controller.select_account(payload)

    @app.get(f"{HELPER_PATH_PREFIX}preview")
    async def take_preview() -> Any:
        preview = await local_controller.take_preview()
        if preview is None:
            return JSONResponse(status_code=409, content={"status": "NOT_READY"})
        return {"status": "OK", "preview": preview}

    @app.post(f"{HELPER_PATH_PREFIX}preview/ack")
    async def acknowledge_preview() -> dict[str, str]:
        return await local_controller.acknowledge_preview()

    @app.post(f"{HELPER_PATH_PREFIX}close")
    async def close() -> dict[str, Any]:
        return await local_controller.close()

    return app


__all__ = [
    "HELPER_PATH_PREFIX",
    "LOCAL_HOST",
    "LOCAL_PORT",
    "NONCE_HEADER",
    "canonicalize_browser_origin",
    "create_desktop_helper_app",
]
