"""SightLine backend server application entrypoint."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Environment & logging
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent / ".env")

# Vertex AI SDK auto-reads GOOGLE_API_KEY / GEMINI_API_KEY from env.
# When VERTEXAI=TRUE, this conflicts with project/location (mutually exclusive).
# Move the API key to a SDK-invisible env var so sub-agents can still read it.
if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").upper() == "TRUE":
    _api_key = os.environ.pop("GOOGLE_API_KEY", "") or os.environ.pop("GEMINI_API_KEY", "")
    os.environ.pop("GEMINI_API_KEY", None)
    os.environ.pop("GOOGLE_API_KEY", None)
    if _api_key:
        os.environ["_GOOGLE_AI_API_KEY"] = _api_key

# Structured JSON logging for Cloud Run; human-readable locally.
# Cloud Run sets K_SERVICE automatically — use it to detect production.
if os.environ.get("K_SERVICE"):

    class _JsonFormatter(logging.Formatter):
        """One JSON object per line — parsed natively by Cloud Logging."""

        def format(self, record: logging.LogRecord) -> str:
            entry: dict = {
                "severity": record.levelname,
                "message": record.getMessage(),
                "time": datetime.fromtimestamp(
                    record.created,
                    tz=timezone.utc,
                ).isoformat(),
                "logger": record.name,
            }
            if record.exc_info and record.exc_info[0]:
                entry["exception"] = self.formatException(record.exc_info)
            return json.dumps(entry)

    _handler = logging.StreamHandler()
    _handler.setFormatter(_JsonFormatter())
    logging.root.addHandler(_handler)
    logging.root.setLevel(logging.INFO)
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

from api.routers import face, health, profile, websocket
from app_globals import PORT

app = FastAPI(title="SightLine Backend", version="0.3.0")

_ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(face.router)
app.include_router(profile.router)
app.include_router(websocket.router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
