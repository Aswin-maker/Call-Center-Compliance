import os
import asyncio
import time
import uuid
import logging
import secrets
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

if __package__:
    from .nlp import analyze_compliance
    from .stt import process_audio
else:
    from nlp import analyze_compliance
    from stt import process_audio

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("call_center_compliance")

MAX_REQUEST_BYTES = int(os.getenv("MAX_REQUEST_BYTES", str(15 * 1024 * 1024)))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "180"))

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

app = FastAPI(
    title="Call Center Compliance API",
    version="0.1.0",
    description="FastAPI scaffold for call center compliance workflows.",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_configured_api_key() -> str:
    # Support both names to reduce config mistakes.
    load_dotenv(override=False)
    return (os.getenv("API_KEY") or os.getenv("X_API_KEY") or "").strip()


@app.on_event("startup")
async def startup_validation() -> None:
    if not get_configured_api_key():
        LOGGER.error(
            "API key is not configured. Set API_KEY in environment or .env file. "
            "Requests to protected endpoints will fail."
        )
    else:
        LOGGER.info("API key authentication is configured")


@app.middleware("http")
async def request_context_and_limits(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id

    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_REQUEST_BYTES:
                LOGGER.warning(
                    "Request rejected: request_id=%s path=%s content_length=%s max=%s",
                    request_id,
                    request.url.path,
                    content_length,
                    MAX_REQUEST_BYTES,
                )
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request payload too large", "request_id": request_id},
                )
        except ValueError:
            LOGGER.debug("Invalid content-length header for request_id=%s", request_id)

    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        LOGGER.exception("Unhandled error in middleware for request_id=%s", request_id)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id},
        )

    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["x-request-id"] = request_id
    response.headers["x-process-time-ms"] = f"{duration_ms:.2f}"

    LOGGER.info(
        "Request completed: request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    openapi_schema["openapi"] = "3.0.3"
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/", include_in_schema=False)
def root() -> HTMLResponse:
    return HTMLResponse(
        content="""
        <html>
            <head><title>Call Center Compliance API</title></head>
            <body style=\"font-family: Arial, sans-serif; margin: 40px;\">
                <h1>Call Center Compliance API</h1>
                <p>API is running successfully.</p>
                <p>Open API Docs: <a href=\"/docs\">/docs</a></p>
            </body>
        </html>
        """
    )


class CallAnalyticsRequest(BaseModel):
    language: str = Field(..., min_length=2, max_length=5)
    audioFormat: str = Field(..., min_length=3, max_length=10)
    audioBase64: str = Field(..., min_length=10)


class SopValidationResponse(BaseModel):
    greeting: bool
    identification: bool
    problemStatement: bool
    solutionOffering: bool
    closing: bool
    complianceScore: float
    adherenceStatus: Literal["FOLLOWED", "NOT_FOLLOWED"]
    explanation: str


class AnalyticsResponse(BaseModel):
    paymentPreference: Literal["EMI", "FULL_PAYMENT", "PARTIAL_PAYMENT", "DOWN_PAYMENT"]
    rejectionReason: Literal[
        "HIGH_INTEREST",
        "BUDGET_CONSTRAINTS",
        "ALREADY_PAID",
        "NOT_INTERESTED",
        "NONE",
    ]
    sentiment: Literal["Positive", "Negative", "Neutral"]


class CallAnalyticsSuccessResponse(BaseModel):
    status: Literal["success"]
    language: str
    transcript: str
    summary: str
    sop_validation: SopValidationResponse
    analytics: AnalyticsResponse
    keywords: list[str]


def verify_api_key(request: Request, x_api_key: str | None = Security(api_key_header)) -> None:
    configured_api_key = get_configured_api_key()
    if not configured_api_key:
        LOGGER.error("API key validation failed because API_KEY is not configured")
        raise HTTPException(status_code=503, detail="Authentication is not configured on server")

    raw_header = request.headers.get("x-api-key")
    incoming = (x_api_key or raw_header or "").strip()

    if not incoming:
        LOGGER.warning("Missing x-api-key header: request_id=%s path=%s", getattr(request.state, "request_id", ""), request.url.path)
        raise HTTPException(status_code=401, detail="Missing x-api-key header")

    if incoming.lower().startswith("bearer "):
        incoming = incoming[7:].strip()

    if not secrets.compare_digest(incoming, configured_api_key):
        LOGGER.warning("Invalid x-api-key: request_id=%s path=%s", getattr(request.state, "request_id", ""), request.url.path)
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", "")
    LOGGER.warning(
        "HTTPException: request_id=%s path=%s status=%s detail=%s",
        request_id,
        request.url.path,
        exc.status_code,
        exc.detail,
    )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail, "request_id": request_id})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "")
    LOGGER.exception(
        "Unhandled exception: request_id=%s method=%s path=%s",
        request_id,
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal processing failure", "request_id": request_id},
    )


@app.post(
    "/api/call-analytics",
    response_model=CallAnalyticsSuccessResponse,
    dependencies=[Security(verify_api_key)],
)
async def call_analytics(
    payload: CallAnalyticsRequest,
    request: Request,
) -> CallAnalyticsSuccessResponse:
    request_id = getattr(request.state, "request_id", "")

    if not payload.language.strip():
        raise HTTPException(status_code=400, detail="language must not be empty")

    normalized_language = payload.language.strip().lower()
    if normalized_language not in {"en", "ta", "hi"}:
        raise HTTPException(status_code=400, detail="language must be en, ta, or hi")

    if payload.audioFormat.strip().lower() != "mp3":
        raise HTTPException(status_code=400, detail="audioFormat must be mp3")

    if not payload.audioBase64.strip():
        raise HTTPException(status_code=400, detail="audioBase64 must not be empty")

    async def _run_pipeline() -> dict:
        transcript_text = await run_in_threadpool(process_audio, payload.audioBase64)
        compliance_result = await run_in_threadpool(analyze_compliance, transcript_text)
        return compliance_result

    try:
        LOGGER.info("Call analytics started: request_id=%s language=%s", request_id, normalized_language)

        compliance = await asyncio.wait_for(
            _run_pipeline(),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        LOGGER.info(
            "Call analytics completed: request_id=%s transcript_chars=%s keywords=%s",
            request_id,
            len(compliance.get("transcript", "")),
            len(compliance.get("keywords", [])),
        )

        return CallAnalyticsSuccessResponse(
            status="success",
            language=compliance["language"],
            transcript=compliance["transcript"],
            summary=compliance["summary"],
            sop_validation=compliance["sop_validation"],
            analytics=compliance["analytics"],
            keywords=compliance["keywords"],
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except asyncio.TimeoutError as exc:
        LOGGER.error("Processing timeout: request_id=%s timeout_s=%s", request_id, REQUEST_TIMEOUT_SECONDS)
        raise HTTPException(status_code=504, detail="Audio processing timeout") from exc
    except MemoryError as exc:
        LOGGER.exception("Memory pressure during processing: request_id=%s", request_id)
        raise HTTPException(status_code=503, detail="Server is temporarily overloaded") from exc
    except Exception as exc:
        LOGGER.exception("Unexpected processing failure: request_id=%s", request_id)
        raise HTTPException(status_code=500, detail="Internal processing failure") from exc
