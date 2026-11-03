import logging
import os
import uuid
from contextlib import asynccontextmanager

import requests
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Literal

from .graph import (
    connection_pool,
    resume_travel_agent,
    run_travel_agent,
    travel_graph,
)
from .auth import ProtectedUser
from .auth_routes import router as auth_router
from .database import engine, get_db
from sqlalchemy.orm import Session
from .trip_routes import router as trips_router
from .trip_service import (
    ensure_user_preferences,
    get_trip_by_owned_thread,
    preference_context,
    record_workflow_update,
    save_workflow_trip,
)

# This is kept from the original project to allow the existing synchronous
# agent functions to call async MCP helpers inside FastAPI.
import nest_asyncio

nest_asyncio.apply()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application):
    yield
    connection_pool.close()
    engine.dispose()


app = FastAPI(
    title="TripMate AI",
    description=(
        "LangGraph Multi-Agent Travel Planner with Supervisor, Guardrails, "
        "Human-in-the-Loop, and FastAPI Frontend"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Configure CORS
origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://localhost:3000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(trips_router)


@app.middleware("http")
async def log_api_requests(request, call_next):
    if request.url.path.startswith("/api/"):
        print(f"API request: {request.method} {request.url.path}", flush=True)

    response = await call_next(request)

    if request.url.path.startswith("/api/"):
        print(
            f"API response: {request.method} {request.url.path} "
            f"{response.status_code}",
            flush=True,
        )

    return response


class TravelRequest(BaseModel):
    message: str = Field(max_length=2000)
    thread_id: str | None = None
    language: Literal["en", "hi", "hinglish"] | None = None


class ApprovalRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    approved: bool
    feedback: str = Field(default="", max_length=1000)
    language: Literal["en", "hi", "hinglish"] | None = None


MAX_AUDIO_BYTES = 20 * 1024 * 1024
ALLOWED_AUDIO_TYPES = {
    "audio/webm": {".webm"},
    "audio/ogg": {".ogg", ".oga"},
    "audio/wav": {".wav"},
    "audio/x-wav": {".wav"},
    "audio/mpeg": {".mp3"},
    "audio/mp3": {".mp3"},
    "audio/mp4": {".mp4", ".m4a"},
    "audio/x-m4a": {".m4a"},
    "audio/aac": {".aac"},
    "audio/flac": {".flac"},
}


def _validate_audio_payload(
    filename: str,
    content_type: str,
    content: bytes,
) -> None:
    suffix = os.path.splitext(filename.lower())[1]
    allowed_extensions = ALLOWED_AUDIO_TYPES.get(
        content_type.split(";", 1)[0].strip().lower()
    )
    if not allowed_extensions or suffix not in allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail="Unsupported audio format. Upload WebM, Ogg, WAV, MP3, MP4/M4A, AAC, or FLAC audio.",
        )
    if len(content) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Audio upload is too large. Maximum size is 20 MB.",
        )
    if not content:
        raise HTTPException(status_code=400, detail="The audio recording is empty.")

    signatures = {
        ".webm": content.startswith(b"\x1a\x45\xdf\xa3"),
        ".ogg": content.startswith(b"OggS"),
        ".oga": content.startswith(b"OggS"),
        ".wav": content.startswith(b"RIFF") and content[8:12] == b"WAVE",
        ".mp3": content.startswith(b"ID3")
        or (
            len(content) >= 2
            and content[0] == 0xFF
            and content[1] & 0xE0 == 0xE0
        ),
        ".mp4": len(content) >= 8 and content[4:8] == b"ftyp",
        ".m4a": len(content) >= 8 and content[4:8] == b"ftyp",
        ".aac": len(content) >= 2 and content[0] == 0xFF and content[1] & 0xF6 == 0xF0,
        ".flac": content.startswith(b"fLaC"),
    }
    if not signatures.get(suffix, False):
        raise HTTPException(
            status_code=415,
            detail="The uploaded file does not match its declared audio format.",
        )


@app.get("/")
async def home():
    print("api main hit", flush=True)
    return {"status": "ok", "message": "TripMate AI API is running"}


@app.post("/api/travel")
async def travel_planner(
    request_data: TravelRequest,
    user: ProtectedUser,
    db: Session = Depends(get_db),
):
    request_id = str(uuid.uuid4())
    try:
        user_message = request_data.message.strip()

        if not user_message:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Message cannot be empty.",
                    "request_id": request_id,
                },
            )

        preferences = preference_context(user)
        result = run_travel_agent(
            user_input=user_message,
            preferred_language=(
                request_data.language
                or str(preferences.get("preferred_language") or "en")
            ),
            user_preferences=preferences,
        )
        trip = save_workflow_trip(db, user, user_message, result)
        if trip is not None:
            db.commit()
            db.refresh(trip)
            result["trip_id"] = str(trip.id)
        return JSONResponse(
            content={
                "success": True,
                **result,
            }
        )

    except HTTPException:
        raise
    except Exception:
        db.rollback()
        logger.exception("Travel request failed request_id=%s", request_id)

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Something went wrong. Please try again.",
                "request_id": request_id,
            },
        )


@app.post("/api/travel/approve")
async def approve_travel_plan(
    request_data: ApprovalRequest,
    user: ProtectedUser,
    db: Session = Depends(get_db),
):
    request_id = str(uuid.uuid4())
    try:
        if not request_data.approved and not request_data.feedback.strip():
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Please provide revision feedback when rejecting the draft.",
                    "request_id": request_id,
                },
            )

        trip = get_trip_by_owned_thread(db, user.id, request_data.thread_id)
        if trip is None:
            raise HTTPException(status_code=404, detail="Travel session not found.")

        config = {"configurable": {"thread_id": trip.thread_id}}
        snapshot = travel_graph.get_state(config)
        if "human_approval" not in snapshot.next:
            return JSONResponse(
                status_code=409,
                content={
                    "success": False,
                    "error": "This travel plan is not waiting for approval.",
                    "request_id": request_id,
                },
            )

        preferences = preference_context(user)
        result = resume_travel_agent(
            thread_id=request_data.thread_id,
            approved=request_data.approved,
            feedback=request_data.feedback,
            preferred_language=(
                request_data.language
                or str(preferences.get("preferred_language") or "en")
            ),
        )
        record_workflow_update(
            db,
            trip,
            user,
            result,
            feedback=request_data.feedback,
        )
        db.commit()
        result["trip_id"] = str(trip.id)

        return JSONResponse(
            content={
                "success": True,
                **result,
            }
        )

    except HTTPException:
        raise
    except Exception:
        db.rollback()
        logger.exception("Travel approval failed request_id=%s", request_id)

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Something went wrong. Please try again.",
                "request_id": request_id,
            },
        )


@app.post("/api/voice/transcribe")
def transcribe_voice(
    user: ProtectedUser,
    audio: UploadFile = File(...),
    language: str = Form(default="auto"),
):
    request_id = str(uuid.uuid4())
    if language not in {"auto", "en", "hi", "hinglish"}:
        raise HTTPException(status_code=422, detail="Unsupported transcription language.")

    content = audio.file.read(MAX_AUDIO_BYTES + 1)
    _validate_audio_payload(
        audio.filename or "",
        audio.content_type or "",
        content,
    )
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Voice transcription is not configured. Text input is still available.",
        )

    suffix = os.path.splitext((audio.filename or "").lower())[1]
    request_data = {
        "model": "whisper-large-v3-turbo",
        "response_format": "verbose_json",
    }
    if language in {"en", "hi"}:
        request_data["language"] = language

    try:
        upstream = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={
                "file": (
                    f"voice-recording{suffix}",
                    content,
                    audio.content_type,
                )
            },
            data=request_data,
            timeout=45,
        )
        upstream.raise_for_status()
        transcription = upstream.json()
    except requests.Timeout:
        logger.warning("Voice transcription timed out request_id=%s", request_id)
        raise HTTPException(
            status_code=504,
            detail="Voice transcription timed out. Please try again or type your request.",
        )
    except requests.HTTPError as exc:
        status_code = (
            503
            if exc.response is not None and exc.response.status_code == 429
            else 502
        )
        logger.warning(
            "Voice transcription provider failed request_id=%s status=%s",
            request_id,
            exc.response.status_code if exc.response is not None else "unknown",
        )
        raise HTTPException(
            status_code=status_code,
            detail="Voice transcription is temporarily unavailable. Please try again or type your request.",
        )
    except (requests.RequestException, ValueError):
        logger.exception("Voice transcription failed request_id=%s", request_id)
        raise HTTPException(
            status_code=502,
            detail="Voice transcription is temporarily unavailable. Please try again or type your request.",
        )

    if not isinstance(transcription, dict):
        raise HTTPException(
            status_code=502,
            detail="The transcription service returned an invalid response. Please try again.",
        )
    transcript = transcription.get("text")
    if not isinstance(transcript, str) or not transcript.strip():
        raise HTTPException(
            status_code=502,
            detail="No speech could be recognized. Please try again or type your request.",
        )
    detected_language = transcription.get("language")
    return {
        "text": transcript.strip(),
        "detected_language": (
            detected_language if isinstance(detected_language, str) else None
        ),
        "request_id": request_id,
    }


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "TripMate AI API is running",
        "features": [
            "supervisor_agent",
            "input_guardrail",
            "human_in_the_loop",
            "multilingual_voice_transcription",
        ],
    }


@app.get("/ready")
def readiness_check():
    """Report ready only when the database/checkpointer can be reached."""
    try:
        with connection_pool.connection() as connection:
            connection.execute("SELECT 1").fetchone()
        return {"status": "ready", "database": "ok"}
    except Exception:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "database": "unavailable"},
        )


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})


if __name__ == "__main__":
    uvicorn.run(
        "backend.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
