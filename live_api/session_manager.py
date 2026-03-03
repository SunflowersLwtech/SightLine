"""Session manager for SightLine Live API connections.

Manages session state, resumption handles, and RunConfig construction
for the Gemini Live API via Google ADK.

Phase 3 additions:
- Firestore UserProfile loading (async, with fallback to defaults)
- Per-session face library cache tracking

Phase 4 additions:
- 3-tier session service fallback: VertexAi → Database → InMemory (depends on AGENT_ENGINE_ID at runtime)
"""

import logging
import os
import time
from typing import Optional

from google.adk.agents.run_config import RunConfig, StreamingMode
from google.genai import types

from lod.models import EphemeralContext, SessionContext, UserProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session service factory (SL-70)
# ---------------------------------------------------------------------------

AGENT_ENGINE_ID = os.getenv("AGENT_ENGINE_ID", "")


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse bool-like environment values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def create_session_service():
    """Create the session service, preferring VertexAiSessionService.

    Uses VertexAiSessionService whenever the SDK provides it. Falls back
    gracefully for local development.
    """
    # Read env at call time because server.py loads .env after imports.
    agent_engine_id = os.getenv("AGENT_ENGINE_ID", "").strip() or None
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "sightline-hackathon")
    location = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
    use_vertex = _env_flag("GOOGLE_GENAI_USE_VERTEXAI", default=False)

    # VertexAiSessionService is intentionally NOT used.
    # It cannot handle client-generated UUID session IDs (returns 400 instead
    # of None for unknown sessions, preventing ADK create_session fallback).
    # Session continuity is handled by Gemini's session_resumption_handle,
    # not by the ADK session service.  All business data lives in Firestore.
    # GOOGLE_GENAI_USE_VERTEXAI=TRUE still routes the Live API model connection
    # through Vertex AI (fixing audio stuttering), which is separate from the
    # session service.
    if use_vertex:
        logger.info(
            "VERTEXAI=TRUE (Live API via Vertex AI); session service = InMemory"
        )
    else:
        logger.info(
            "VERTEXAI=FALSE (Live API via Google AI); session service = InMemory"
        )

    # Both local and Cloud Run use in-memory session service.
    # Session continuity is handled by Gemini's session_resumption_handle,
    # not by the ADK session service. All business data lives in Firestore.
    from google.adk.sessions import InMemorySessionService as _InMemory

    logger.info("Using in-memory session service")
    return _InMemory()

# ---------------------------------------------------------------------------
# LOD-driven VAD presets (SL-36)
# ---------------------------------------------------------------------------

# NOTE: Server-side VAD is enabled with conservative sensitivity.
# Client-side RMS gating (silence during model speech) acts as AEC
# to prevent echo residual from triggering server VAD.
LOD_VAD_PRESETS: dict[int, dict] = {
    1: {
        "voice_name": "Aoede",
        "start_sensitivity": types.StartSensitivity.START_SENSITIVITY_LOW,
        "end_sensitivity": types.EndSensitivity.END_SENSITIVITY_LOW,
        "silence_duration_ms": 800,
        "prefix_padding_ms": 250,
    },
    2: {
        "voice_name": "Aoede",
        "start_sensitivity": types.StartSensitivity.START_SENSITIVITY_LOW,
        "end_sensitivity": types.EndSensitivity.END_SENSITIVITY_LOW,
        "silence_duration_ms": 1500,
        "prefix_padding_ms": 300,
    },
    3: {
        "voice_name": "Aoede",
        "start_sensitivity": types.StartSensitivity.START_SENSITIVITY_LOW,
        "end_sensitivity": types.EndSensitivity.END_SENSITIVITY_LOW,
        "silence_duration_ms": 1500,
        "prefix_padding_ms": 300,
    },
}


def _enum_label(value) -> str:
    """Convert SDK enum values into stable log/debug labels."""
    if value is None:
        return "UNSPECIFIED"
    if hasattr(value, "name"):
        return str(value.name)
    return str(value).split(".")[-1]


def get_lod_vad_preset(lod: int) -> dict:
    """Return a copy of the LOD-specific VAD preset."""
    return dict(LOD_VAD_PRESETS.get(lod, LOD_VAD_PRESETS[2]))


def supports_runtime_vad_reconfiguration() -> tuple[bool, str]:
    """Whether ADK transport can hot-update VAD config mid-session.

    google-adk 1.25.x LiveRequestQueue currently only supports:
    content, realtime blobs, and activity start/end signals.
    """
    return (
        False,
        "LiveRequestQueue does not expose realtime_input_config updates mid-session.",
    )


def build_vad_runtime_update_payload(lod: int) -> dict:
    """Build a serializable VAD payload for logging/debug contracts."""
    preset = get_lod_vad_preset(lod)
    return {
        "lod": lod,
        "start_of_speech_sensitivity": _enum_label(preset.get("start_sensitivity")),
        "end_of_speech_sensitivity": _enum_label(preset.get("end_sensitivity")),
        "prefix_padding_ms": int(preset.get("prefix_padding_ms", 200)),
        "silence_duration_ms": int(preset.get("silence_duration_ms", 800)),
    }


def build_vad_runtime_update_message(lod: int) -> str:
    """Build an internal control message injected on LOD transitions."""
    payload = build_vad_runtime_update_payload(lod)
    return (
        "[VAD UPDATE]\n"
        "Internal control sync. Keep this as configuration context only.\n"
        f"- LOD: {payload['lod']}\n"
        f"- start_of_speech_sensitivity: {payload['start_of_speech_sensitivity']}\n"
        f"- end_of_speech_sensitivity: {payload['end_of_speech_sensitivity']}\n"
        f"- prefix_padding_ms: {payload['prefix_padding_ms']}\n"
        f"- silence_duration_ms: {payload['silence_duration_ms']}\n"
        "Do not narrate this block to the user."
    )

# ---------------------------------------------------------------------------
# Firestore client (lazy)
# ---------------------------------------------------------------------------

_firestore_client = None


def _get_firestore():
    """Lazily initialize the Firestore client."""
    global _firestore_client
    if _firestore_client is None:
        try:
            from google.cloud import firestore
            project = os.getenv("GOOGLE_CLOUD_PROJECT", "sightline-hackathon")
            _firestore_client = firestore.Client(project=project)
        except Exception:
            logger.warning("Firestore client unavailable; using default profiles")
            _firestore_client = False  # Sentinel to avoid retrying
    return _firestore_client if _firestore_client else None


class SessionManager:
    """Manages Live API session state and RunConfig construction.

    Tracks session resumption handles and per-session context so that
    dropped connections can be transparently resumed.
    """

    _USER_PROFILE_TTL_SEC = 3600.0  # 1 hour

    def __init__(self) -> None:
        self._session_handles: dict[str, str] = {}
        self._session_contexts: dict[str, SessionContext] = {}
        self._user_profiles: dict[str, UserProfile] = {}
        self._user_profile_access_times: dict[str, float] = {}
        self._ephemeral_contexts: dict[str, EphemeralContext] = {}

    # -- RunConfig ----------------------------------------------------------

    # Languages supported by Gemini Live API native audio models.
    # zh-CN/zh-TW are NOT supported; language preference is handled
    # via system instructions in prompt_builder instead.
    _NATIVE_AUDIO_LANGUAGES = {
        "ar-EG", "bn-BD", "nl-NL", "en-IN", "en-US", "fr-FR", "de-DE",
        "hi-IN", "id-ID", "it-IT", "ja-JP", "ko-KR", "mr-IN", "pl-PL",
        "pt-BR", "ro-RO", "ru-RU", "es-ES", "es-US", "ta-IN", "te-IN", "th-TH",
        "tr-TR", "uk-UA", "vi-VN",
    }

    def get_run_config(self, session_id: str, lod: int = 2, language_code: str = "") -> RunConfig:
        """Build a RunConfig for the given session."""
        cached_handle = self._session_handles.get(session_id)

        session_resumption = types.SessionResumptionConfig(
            handle=cached_handle,
        )
        if cached_handle:
            logger.info("Resuming session %s with cached handle", session_id)
        else:
            logger.info("Starting fresh session %s (no cached handle)", session_id)

        vad_preset = get_lod_vad_preset(lod)

        # Only pass language_code if it's supported by the native audio model.
        # Unsupported languages (e.g. zh-CN) are handled via system instructions
        # in prompt_builder.build_user_profile_block() instead.
        effective_lang = None
        if language_code and language_code in self._NATIVE_AUDIO_LANGUAGES:
            effective_lang = language_code

        run_config = RunConfig(
            streaming_mode=StreamingMode.BIDI,
            response_modalities=[types.Modality.AUDIO],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=vad_preset["voice_name"],
                    )
                ),
                language_code=effective_lang,
            ),
            # proactive_audio and affective_dialog disabled:
            # - affective_dialog is correlated with premature turnComplete
            #   (js-genai#707, python-genai#872) — model cuts off mid-sentence
            # - proactive_audio conflicts with affective_dialog
            #   (Google AI Studio mutually excludes them)
            # Re-enable individually after verifying audio stability.
            # proactivity=types.ProactivityConfig(proactive_audio=True),
            # enable_affective_dialog=True,
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            session_resumption=session_resumption,
            context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=100_000,
                sliding_window=types.SlidingWindow(target_tokens=80_000),
            ),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=False,
                    start_of_speech_sensitivity=vad_preset["start_sensitivity"],
                    end_of_speech_sensitivity=vad_preset["end_sensitivity"],
                    prefix_padding_ms=vad_preset["prefix_padding_ms"],
                    silence_duration_ms=vad_preset["silence_duration_ms"],
                ),
                # NO_INTERRUPTION: Gemini's server-side VAD no longer interrupts
                # model audio output.  This eliminates self-interruption caused by
                # echo residual being misdetected as user speech.  True barge-in
                # is handled client-side (iOS RMS + SileroVAD confirmation) —
                # iOS stops playback locally and the server suppresses forwarding
                # until turn_complete.
                activity_handling=types.ActivityHandling.NO_INTERRUPTION,
                turn_coverage=types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,
            ),
        )

        return run_config

    # -- Session handle cache -----------------------------------------------

    def update_handle(self, session_id: str, handle: str) -> None:
        """Cache a session resumption handle."""
        self._session_handles[session_id] = handle
        logger.debug("Cached resumption handle for session %s", session_id)

    def get_handle(self, session_id: str) -> Optional[str]:
        """Retrieve a cached resumption handle."""
        return self._session_handles.get(session_id)

    # -- Per-session context ------------------------------------------------

    def get_session_context(self, session_id: str) -> SessionContext:
        """Get or create the SessionContext for this session."""
        if session_id not in self._session_contexts:
            self._session_contexts[session_id] = SessionContext()
        return self._session_contexts[session_id]

    async def load_user_profile(self, user_id: str) -> UserProfile:
        """Load UserProfile from Firestore, falling back to defaults.

        Caches the result so subsequent calls for the same user_id
        return the cached profile without hitting Firestore again.
        """
        if user_id in self._user_profiles:
            self._user_profile_access_times[user_id] = time.monotonic()
            return self._user_profiles[user_id]

        profile = UserProfile.default()
        profile.user_id = user_id

        db = _get_firestore()
        if db:
            try:
                doc = db.collection("user_profiles").document(user_id).get()
                if doc.exists:
                    profile = UserProfile.from_firestore(doc.to_dict(), user_id=user_id)
                    logger.info("Loaded UserProfile from Firestore for user %s", user_id)
                else:
                    logger.info("No Firestore profile for user %s; using defaults", user_id)
            except Exception:
                logger.exception("Failed to load profile for user %s; using defaults", user_id)

        self._user_profiles[user_id] = profile
        self._user_profile_access_times[user_id] = time.monotonic()
        return profile

    def invalidate_user_profile(self, user_id: str) -> None:
        """Remove cached profile so next load_user_profile fetches fresh data."""
        self._user_profiles.pop(user_id, None)
        self._user_profile_access_times.pop(user_id, None)

    def get_user_profile(self, user_id: str) -> UserProfile:
        """Get cached UserProfile (sync). Use load_user_profile for initial load."""
        if user_id not in self._user_profiles:
            self._user_profiles[user_id] = UserProfile.default()
            self._user_profiles[user_id].user_id = user_id
        self._user_profile_access_times[user_id] = time.monotonic()
        return self._user_profiles[user_id]

    def evict_stale_profiles(self) -> int:
        """Remove cached profiles not accessed within _USER_PROFILE_TTL_SEC.

        Returns the number of evicted profiles.
        """
        now = time.monotonic()
        stale = [
            uid for uid, ts in self._user_profile_access_times.items()
            if (now - ts) > self._USER_PROFILE_TTL_SEC
        ]
        for uid in stale:
            self._user_profiles.pop(uid, None)
            self._user_profile_access_times.pop(uid, None)
        if stale:
            logger.info("Evicted %d stale user profile(s)", len(stale))
        return len(stale)

    def get_ephemeral_context(self, session_id: str) -> EphemeralContext:
        """Get or create the latest EphemeralContext for this session."""
        if session_id not in self._ephemeral_contexts:
            self._ephemeral_contexts[session_id] = EphemeralContext()
        return self._ephemeral_contexts[session_id]

    def update_ephemeral_context(self, session_id: str, ctx: EphemeralContext) -> None:
        """Store the latest ephemeral context snapshot."""
        self._ephemeral_contexts[session_id] = ctx

    # -- Cleanup ------------------------------------------------------------

    def remove_session(self, session_id: str) -> None:
        """Remove all cached state for a session and evict stale profiles."""
        self._session_handles.pop(session_id, None)
        self._session_contexts.pop(session_id, None)
        self._ephemeral_contexts.pop(session_id, None)
        self.evict_stale_profiles()
        logger.debug("Removed session state for %s", session_id)
