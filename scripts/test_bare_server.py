"""Bare-bones test server — NO tools, NO context injection, NO memory.

Purpose: Isolate whether audio truncation is caused by our code complexity
or by a Gemini server-side bug.

Usage:
    conda activate sightline
    python test_bare_server.py

Then connect iOS client as usual (ws://Lius-MacBook-Air.local:8100/ws/...).
If audio STILL cuts off → it's Google's server-side bug.
If audio is FINE → our context injection volume is the trigger.
"""

import asyncio
import base64
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from google.adk.agents import Agent
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.genai import types
from starlette.websockets import WebSocketState

load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bare_test")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# --- Force Vertex AI backend for GA model ---
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
os.environ["GOOGLE_CLOUD_PROJECT"] = os.getenv("GOOGLE_CLOUD_PROJECT", "sightline-hackathon")
os.environ["GOOGLE_CLOUD_LOCATION"] = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
# Vertex AI uses ADC, not API keys — remove them to avoid conflict
os.environ.pop("GOOGLE_API_KEY", None)
os.environ.pop("GEMINI_API_KEY", None)

LIVE_MODEL = "gemini-live-2.5-flash-native-audio"  # Vertex AI GA model
PORT = int(os.getenv("PORT", "8100"))

# ---------------------------------------------------------------------------
# Minimal agent — NO tools, NO sub-agents, short prompt
# ---------------------------------------------------------------------------

MINIMAL_PROMPT = """\
You are a friendly assistant. When the user speaks to you, respond naturally \
and in complete sentences. If the user asks you to describe something, give \
a detailed response of at least 3-4 sentences. Never stop mid-sentence.
"""

agent = Agent(
    model=LIVE_MODEL,
    name="bare_test_agent",
    instruction=MINIMAL_PROMPT,
    tools=[],  # NO tools
)

from google.adk.sessions import InMemorySessionService

runner = Runner(
    agent=agent,
    app_name="sightline_bare_test",
    session_service=InMemorySessionService(),
    auto_create_session=True,
)

# ---------------------------------------------------------------------------
# Minimal RunConfig — NO compression, NO affective, NO proactive
# ---------------------------------------------------------------------------


def get_bare_run_config() -> RunConfig:
    """Absolutely minimal Live API config."""
    return RunConfig(
        streaming_mode=StreamingMode.BIDI,
        response_modalities=[types.Modality.AUDIO],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Aoede",
                )
            ),
        ),
        # --- Everything below is deliberately OFF ---
        # NO proactive_audio
        # NO enable_affective_dialog
        # NO context_window_compression
        # NO session_resumption
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=False,
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
                prefix_padding_ms=200,
                silence_duration_ms=800,
            ),
            activity_handling=types.ActivityHandling.NO_INTERRUPTION,
            turn_coverage=types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,
        ),
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="SightLine Bare Test", version="0.0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "mode": "BARE_TEST", "model": LIVE_MODEL}


@app.websocket("/ws/{user_id}/{session_id}")
async def ws_endpoint(websocket: WebSocket, user_id: str, session_id: str):
    await websocket.accept()
    logger.info("iOS connected: user=%s session=%s", user_id, session_id)

    await websocket.send_json({"type": "session_ready"})

    live_queue = LiveRequestQueue()
    run_config = get_bare_run_config()

    live_events = runner.run_live(
        session_id=session_id,
        user_id=user_id,
        live_request_queue=live_queue,
        run_config=run_config,
    )

    async def upstream():
        """iOS → Gemini: forward audio only, ignore everything else."""
        try:
            while True:
                msg = await websocket.receive()

                if "bytes" in msg:
                    raw = msg["bytes"]
                    if len(raw) < 2:
                        continue
                    magic = raw[0]
                    payload = raw[1:]
                    if magic == 0x01:  # audio
                        live_queue.send_realtime(
                            types.Blob(
                                data=payload,
                                mime_type="audio/pcm;rate=16000",
                            )
                        )
                    # Ignore images (0x02) — bare test
                    continue

                if "text" in msg:
                    try:
                        data = json.loads(msg["text"])
                    except json.JSONDecodeError:
                        continue
                    t = data.get("type")
                    if t == "audio":
                        audio = base64.b64decode(data["data"])
                        live_queue.send_realtime(
                            types.Blob(
                                data=audio,
                                mime_type="audio/pcm;rate=16000",
                            )
                        )
                    elif t == "activity_start":
                        live_queue.send_activity_start()
                    elif t == "activity_end":
                        live_queue.send_activity_end()
                    # Ignore telemetry, gestures, etc.
        except WebSocketDisconnect:
            logger.info("iOS disconnected (upstream)")
        except Exception as e:
            logger.exception("Upstream error: %s", e)
        finally:
            live_queue.close()

    async def downstream():
        """Gemini → iOS: forward audio + transcript only."""
        turn_count = 0
        try:
            async for event in live_events:
                if websocket.client_state != WebSocketState.CONNECTED:
                    break

                # Audio chunks
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.inline_data and part.inline_data.data:
                            audio = part.inline_data.data
                            if isinstance(audio, str):
                                audio = base64.b64decode(audio)
                            await websocket.send_bytes(audio)

                # Agent transcript
                if (
                    hasattr(event, "output_transcription")
                    and event.output_transcription
                    and event.output_transcription.text
                ):
                    await websocket.send_json(
                        {
                            "type": "transcript",
                            "text": event.output_transcription.text,
                            "role": "agent",
                        }
                    )

                # Turn complete — log it for diagnosis
                if getattr(event, "turn_complete", False):
                    turn_count += 1
                    interrupted = getattr(event, "interrupted", None)
                    logger.warning(
                        ">>> TURN COMPLETE #%d | interrupted=%s",
                        turn_count,
                        interrupted,
                    )

                # Session resumption handle
                if (
                    hasattr(event, "session_resumption_update")
                    and event.session_resumption_update
                ):
                    handle = event.session_resumption_update.new_handle
                    if handle:
                        await websocket.send_json(
                            {"type": "session_resumption", "handle": handle}
                        )

        except WebSocketDisconnect:
            logger.info("iOS disconnected (downstream)")
        except Exception as e:
            logger.exception("Downstream error: %s", e)

    up_task = asyncio.create_task(upstream())
    down_task = asyncio.create_task(downstream())

    done, pending = await asyncio.wait(
        [up_task, down_task], return_when=asyncio.FIRST_COMPLETED
    )
    for t in pending:
        t.cancel()

    if websocket.client_state == WebSocketState.CONNECTED:
        await websocket.close()
    logger.info("Session %s closed", session_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logger.info("=== BARE TEST SERVER === Model: %s | Port: %d", LIVE_MODEL, PORT)
    logger.info("No tools, no context injection, no memory, no vision/OCR")
    logger.info("If audio still cuts off → Google server-side bug confirmed")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
