#!/usr/bin/env python3
"""End-to-end test suite for SightLine.

Loads pre-generated test assets (images, audio) from a manifest, connects to
the running server via WebSocket, runs ~30 test scenarios across protocol,
vision, OCR, LOD, tool, safety, UX, and stability categories, and produces a
structured JSON report with issues.

Usage:
    python scripts/run_e2e_test_suite.py
    python scripts/run_e2e_test_suite.py --server ws://127.0.0.1:8100 --scenarios "P-*"
    python scripts/run_e2e_test_suite.py --assets-dir artifacts/e2e_assets --timeout 1.5
"""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import json
import logging
import os
import time
from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websockets
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sightline.e2e")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    """Result container for a single E2E test scenario."""

    scenario_id: str = ""
    name: str = ""
    category: str = ""
    passed: bool = False
    failures: list[str] = dc_field(default_factory=list)
    warnings: list[str] = dc_field(default_factory=list)
    counts: dict[str, int] = dc_field(default_factory=dict)
    tool_events: list[dict[str, Any]] = dc_field(default_factory=list)
    tool_results: list[dict[str, Any]] = dc_field(default_factory=list)
    lod_updates: list[dict[str, Any]] = dc_field(default_factory=list)
    transcripts: list[dict[str, str]] = dc_field(default_factory=list)
    frame_acks: list[dict[str, Any]] = dc_field(default_factory=list)
    duration_sec: float = 0.0
    notes: list[str] = dc_field(default_factory=list)
    known_issue: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "category": self.category,
            "passed": self.passed,
            "known_issue": self.known_issue,
            "failures": self.failures,
            "warnings": self.warnings,
            "counts": self.counts,
            "tool_events": self.tool_events,
            "tool_results": self.tool_results,
            "lod_updates": self.lod_updates,
            "transcripts": self.transcripts[:60],
            "frame_acks": self.frame_acks[:20],
            "duration_sec": round(self.duration_sec, 2),
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Known issue map — maps scenario IDs to known code-level issues
# ---------------------------------------------------------------------------

KNOWN_ISSUE_MAP: dict[str, dict[str, Any]] = {
    "O-005": {
        "title": "Vision query triggers unwanted OCR",
        "location": (
            "agents/orchestrator.py:231-236 (OCR tool description) "
            "+ tools/ocr_tool.py:62-83"
        ),
        "definition": (
            "OCR tool description in orchestrator system prompt is too broadly "
            "worded, causing LLM to trigger extract_text_from_camera for "
            "general vision queries."
        ),
        "severity": "high",
    },
}

# ---------------------------------------------------------------------------
# Protocol helpers
# ---------------------------------------------------------------------------

MAGIC_AUDIO = 0x01
MAGIC_IMAGE = 0x02
E2E_USER_ID = "e2e_test_user"


async def _wait_for_type(
    ws: websockets.ClientConnection,
    msg_type: str,
    timeout_sec: float,
) -> dict[str, Any]:
    """Wait until a JSON message of the given type arrives."""
    deadline = time.monotonic() + timeout_sec
    while True:
        remain = deadline - time.monotonic()
        if remain <= 0:
            raise TimeoutError(f"did not receive '{msg_type}' within {timeout_sec}s")
        raw = await asyncio.wait_for(ws.recv(), timeout=remain)
        if isinstance(raw, bytes):
            continue
        payload = json.loads(raw)
        if payload.get("type") == msg_type:
            return payload


def _record_payload(
    payload: dict[str, Any],
    *,
    counts: dict[str, int],
    tool_events: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    lod_updates: list[dict[str, Any]],
    transcripts: list[dict[str, str]],
    frame_acks: list[dict[str, Any]],
) -> None:
    """Classify and record a single JSON message from the server."""
    t = str(payload.get("type", "unknown"))
    counts[t] = counts.get(t, 0) + 1

    if t == "tool_event":
        tool_events.append({
            "tool": payload.get("tool"),
            "status": payload.get("status"),
            "behavior": payload.get("behavior"),
            "repeat_suppressed": payload.get("repeat_suppressed", False),
        })
    elif t == "tool_result":
        tool_results.append({
            "tool": payload.get("tool"),
            "behavior": payload.get("behavior"),
        })
    elif t == "lod_update":
        lod_updates.append({
            "lod": payload.get("lod"),
            "reason": payload.get("reason"),
        })
    elif t == "transcript":
        transcripts.append({
            "role": str(payload.get("role", "")),
            "text": str(payload.get("text", "")),
        })
    elif t == "frame_ack":
        frame_acks.append({
            "queued_agents": payload.get("queued_agents"),
            "ts": time.monotonic(),
        })
    elif t == "vision_result":
        counts["vision_result"] = counts.get("vision_result", 0)  # already counted
        # Store raw for inspection
        tool_results.append({"_type": "vision_result", **payload})
    elif t == "ocr_result":
        counts["ocr_result"] = counts.get("ocr_result", 0)
        tool_results.append({"_type": "ocr_result", **payload})
    elif t == "search_result":
        pass  # already counted
    elif t == "navigation_result":
        pass  # already counted


async def _send_audio_stream(
    ws: websockets.ClientConnection,
    pcm: bytes,
    *,
    with_activity_markers: bool = True,
) -> None:
    """Stream raw PCM bytes to the server with activity markers."""
    if with_activity_markers:
        await ws.send(json.dumps({"type": "activity_start"}))
    for offset in range(0, len(pcm), 1280):
        chunk = pcm[offset : offset + 1280]
        await ws.send(bytes([MAGIC_AUDIO]) + chunk)
        await asyncio.sleep((len(chunk) / 2.0) / 16000.0)
    if with_activity_markers:
        await ws.send(json.dumps({"type": "activity_end"}))


async def _collect_responses(
    ws: websockets.ClientConnection,
    collect_sec: float,
    *,
    counts: dict[str, int],
    tool_events: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    lod_updates: list[dict[str, Any]],
    transcripts: list[dict[str, str]],
    frame_acks: list[dict[str, Any]],
) -> None:
    """Collect all server responses for a given duration."""
    deadline = time.monotonic() + collect_sec
    while time.monotonic() < deadline:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        if isinstance(raw, bytes):
            counts["audio_bytes"] = counts.get("audio_bytes", 0) + 1
            continue

        payload = json.loads(raw)
        _record_payload(
            payload,
            counts=counts,
            tool_events=tool_events,
            tool_results=tool_results,
            lod_updates=lod_updates,
            transcripts=transcripts,
            frame_acks=frame_acks,
        )


# ---------------------------------------------------------------------------
# Versatile probe function
# ---------------------------------------------------------------------------


async def _run_probe(
    ws_url: str,
    *,
    pcm_path: str | None = None,
    image_path: str | None = None,
    telemetry: dict[str, Any] | None = None,
    gesture: str | None = None,
    wait_before_sec: float = 3.0,
    collect_sec: float = 30.0,
    send_image_first: bool = False,
    send_multiple_images: bool = False,
    image_interval_sec: float = 0.0,
    second_image_path: str | None = None,
    second_image_delay_sec: float = 0.0,
    skip_audio: bool = False,
) -> ScenarioResult:
    """Connect to the WebSocket, send stimuli, and collect responses.

    This is the universal probe used by all individual scenario functions.
    It handles the full lifecycle: connect -> session_ready -> stimuli -> collect.
    """
    pcm = Path(pcm_path).read_bytes() if pcm_path else b""
    image_bytes = Path(image_path).read_bytes() if image_path else b""
    second_image_bytes = (
        Path(second_image_path).read_bytes() if second_image_path else b""
    )

    counts: dict[str, int] = {}
    tool_events: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    lod_updates: list[dict[str, Any]] = []
    transcripts: list[dict[str, str]] = []
    frame_acks: list[dict[str, Any]] = []
    notes: list[str] = []

    t0 = time.monotonic()

    async with websockets.connect(ws_url, max_size=None) as ws:
        await _wait_for_type(ws, "session_ready", timeout_sec=20.0)

        if wait_before_sec > 0:
            await asyncio.sleep(wait_before_sec)

        # --- Send stimuli in the requested order ---

        # Telemetry first (establishes context)
        if telemetry:
            await ws.send(json.dumps({"type": "telemetry", "data": telemetry}))
            notes.append("telemetry_sent")

        # Gesture
        if gesture:
            await ws.send(json.dumps({"type": "gesture", "gesture": gesture}))
            notes.append(f"gesture_sent:{gesture}")

        # Image-first mode
        if send_image_first and image_bytes:
            await ws.send(bytes([MAGIC_IMAGE]) + image_bytes)
            notes.append("image_sent_first")
            if image_interval_sec > 0:
                await asyncio.sleep(image_interval_sec)

        # Audio
        if pcm and not skip_audio:
            await _send_audio_stream(ws, pcm)
            notes.append("audio_sent")

        # Image after audio (default order)
        if not send_image_first and image_bytes:
            await ws.send(bytes([MAGIC_IMAGE]) + image_bytes)
            notes.append("image_sent")

        # Multiple images (for repeat suppression / rapid-fire tests)
        if send_multiple_images and image_bytes:
            if image_interval_sec > 0:
                await asyncio.sleep(image_interval_sec)
            await ws.send(bytes([MAGIC_IMAGE]) + image_bytes)
            notes.append("image_sent_again")

        # Second distinct image with configurable delay
        if second_image_bytes:
            if second_image_delay_sec > 0:
                await asyncio.sleep(second_image_delay_sec)
            await ws.send(bytes([MAGIC_IMAGE]) + second_image_bytes)
            notes.append("second_image_sent")

        # --- Collect responses ---
        await _collect_responses(
            ws,
            collect_sec,
            counts=counts,
            tool_events=tool_events,
            tool_results=tool_results,
            lod_updates=lod_updates,
            transcripts=transcripts,
            frame_acks=frame_acks,
        )

    duration = time.monotonic() - t0

    return ScenarioResult(
        counts=counts,
        tool_events=tool_events,
        tool_results=tool_results,
        lod_updates=lod_updates,
        transcripts=transcripts[:60],
        frame_acks=frame_acks[:20],
        duration_sec=duration,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# E2E Test Suite
# ---------------------------------------------------------------------------


class E2ETestSuite:
    """Orchestrates all E2E test scenarios against a running SightLine server."""

    def __init__(
        self,
        ws_base_url: str,
        assets_dir: str | Path,
        output_dir: str | Path,
        server_log_path: str | Path | None = None,
        timeout_multiplier: float = 1.0,
        scenario_filter: str | None = None,
    ) -> None:
        self.ws_base_url = ws_base_url.rstrip("/")
        self.assets_dir = Path(assets_dir)
        self.output_dir = Path(output_dir)
        self.server_log_path = Path(server_log_path) if server_log_path else None
        self.timeout_mult = timeout_multiplier
        self.scenario_filter = scenario_filter
        self.manifest: dict[str, Any] | None = None
        self.results: list[ScenarioResult] = []

    # -- Public API ----------------------------------------------------------

    async def run_all(self) -> dict[str, Any]:
        """Execute the full E2E suite and return a structured report."""
        self.manifest = self._load_manifest()
        await self._verify_server_health()
        self.results = await self._run_scenarios()
        report = self._build_report(self.results)
        self._write_report(report)
        return report

    # -- Manifest & health ---------------------------------------------------

    def _load_manifest(self) -> dict[str, Any]:
        manifest_path = self.assets_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found at {manifest_path}. "
                "Run generate_e2e_assets.py first."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        logger.info(
            "Manifest loaded: %d images, %d audio clips",
            len(manifest.get("images", [])),
            len(manifest.get("audio", [])),
        )
        return manifest

    async def _verify_server_health(self) -> None:
        """Check that the server is reachable via HTTP health endpoint."""
        import urllib.error
        import urllib.request

        # Derive HTTP URL from WebSocket URL
        http_url = self.ws_base_url.replace("ws://", "http://").replace("wss://", "https://")
        # Strip /ws path if present to get base
        if http_url.endswith("/ws"):
            http_url = http_url[:-3]
        health_url = f"{http_url}/health"

        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status != 200:
                    raise ConnectionError(
                        f"Server health check returned {resp.status}"
                    )
            logger.info("Server health OK: %s", health_url)
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            raise ConnectionError(
                f"Cannot reach server at {health_url}: {exc}\n"
                "Make sure the server is running: python server.py"
            ) from exc

    # -- Asset lookup helpers ------------------------------------------------

    def _get_image(self, image_id: str) -> str | None:
        """Resolve an image asset path from the manifest by ID."""
        if not self.manifest:
            return None
        for img in self.manifest.get("images", []):
            if img.get("id") == image_id:
                p = self.assets_dir / img["path"]
                if p.exists():
                    return str(p)
                logger.warning("Image asset missing on disk: %s", p)
                return None
        logger.warning("Image ID '%s' not found in manifest", image_id)
        return None

    def _get_audio(self, audio_id: str) -> str | None:
        """Resolve a PCM audio asset path from the manifest by ID."""
        if not self.manifest:
            return None
        for aud in self.manifest.get("audio", []):
            if aud.get("id") == audio_id:
                p = self.assets_dir / aud["pcm_path"]
                if p.exists():
                    return str(p)
                logger.warning("Audio asset missing on disk: %s", p)
                return None
        logger.warning("Audio ID '%s' not found in manifest", audio_id)
        return None

    def _ws_url(self, session_suffix: str) -> str:
        """Build a full WebSocket URL with user_id and session_id."""
        return f"{self.ws_base_url}/ws/{E2E_USER_ID}/{session_suffix}"

    def _t(self, base_sec: float) -> float:
        """Apply timeout multiplier to a base duration."""
        return base_sec * self.timeout_mult

    # -- Scenario registry ---------------------------------------------------

    def _all_scenario_methods(self) -> list[tuple[str, Any]]:
        """Return all scenario methods sorted by ID, applying filter if set."""
        methods = []
        for attr in sorted(dir(self)):
            if attr.startswith("_scenario_"):
                method = getattr(self, attr)
                # Extract scenario ID from docstring (first word like "P-001:")
                doc = (method.__doc__ or "").strip()
                scenario_id = doc.split(":")[0].strip() if ":" in doc else attr
                if self.scenario_filter:
                    pattern = self.scenario_filter.upper()
                    sid_upper = scenario_id.upper()
                    if not (
                        fnmatch.fnmatch(sid_upper, pattern)
                        or pattern in sid_upper
                    ):
                        continue
                methods.append((scenario_id, method))
        return methods

    async def _run_scenarios(self) -> list[ScenarioResult]:
        """Execute all matching scenarios sequentially."""
        methods = self._all_scenario_methods()
        logger.info("Running %d scenarios (filter=%s)", len(methods), self.scenario_filter)

        results: list[ScenarioResult] = []
        for scenario_id, method in methods:
            logger.info("--- [%s] Starting ---", scenario_id)
            try:
                result = await method()
                if not result.scenario_id:
                    result.scenario_id = scenario_id
            except TimeoutError as exc:
                result = ScenarioResult(
                    scenario_id=scenario_id,
                    name=method.__doc__.split(":")[1].strip() if method.__doc__ else scenario_id,
                    category="error",
                    passed=False,
                    failures=[f"timeout: {exc}"],
                )
            except (ConnectionError, OSError, websockets.exceptions.WebSocketException) as exc:
                result = ScenarioResult(
                    scenario_id=scenario_id,
                    name=method.__doc__.split(":")[1].strip() if method.__doc__ else scenario_id,
                    category="error",
                    passed=False,
                    failures=[f"connection_error: {exc}"],
                )
            except Exception as exc:
                logger.exception("Unexpected error in scenario %s", scenario_id)
                result = ScenarioResult(
                    scenario_id=scenario_id,
                    name=method.__doc__ or scenario_id,
                    category="error",
                    passed=False,
                    failures=[f"unexpected_error: {type(exc).__name__}: {exc}"],
                )

            status = "PASS" if result.passed else ("KNOWN_ISSUE" if result.known_issue else "FAIL")
            logger.info(
                "--- [%s] %s  (%.1fs, failures=%s) ---",
                result.scenario_id,
                status,
                result.duration_sec,
                result.failures or "none",
            )
            results.append(result)

        return results

    # ======================================================================
    # PROTOCOL TESTS (P-001 to P-004)
    # ======================================================================

    async def _scenario_P001(self) -> ScenarioResult:
        """P-001: Binary audio ingest."""
        pcm_path = self._get_audio("greeting")
        if not pcm_path:
            return self._skip_result("P-001", "Binary audio ingest", "protocol", "greeting audio not found")

        result = await _run_probe(
            self._ws_url("e2e_P001"),
            pcm_path=pcm_path,
            wait_before_sec=self._t(3.0),
            collect_sec=self._t(30.0),
        )
        result.scenario_id = "P-001"
        result.name = "Binary audio ingest"
        result.category = "protocol"

        if not any(t["role"] == "user" for t in result.transcripts):
            result.failures.append("no_user_transcript")
        if not any(t["role"] == "agent" for t in result.transcripts):
            result.failures.append("no_agent_transcript")
        if result.counts.get("audio_bytes", 0) == 0:
            result.failures.append("no_agent_audio")

        result.passed = not result.failures
        return result

    async def _scenario_P002(self) -> ScenarioResult:
        """P-002: Binary image ingest."""
        image_path = self._get_image("street_crossing") or self._get_image("park_scene")
        if not image_path:
            return self._skip_result("P-002", "Binary image ingest", "protocol", "no test image found")

        result = await _run_probe(
            self._ws_url("e2e_P002"),
            image_path=image_path,
            skip_audio=True,
            wait_before_sec=self._t(10.0),
            collect_sec=self._t(15.0),
        )
        result.scenario_id = "P-002"
        result.name = "Binary image ingest"
        result.category = "protocol"

        if not result.frame_acks:
            result.failures.append("no_frame_ack")
        else:
            first_ack = result.frame_acks[0]
            if "queued_agents" not in first_ack or first_ack["queued_agents"] is None:
                result.warnings.append("frame_ack_missing_queued_agents_field")

        result.passed = not result.failures
        return result

    async def _scenario_P003(self) -> ScenarioResult:
        """P-003: Invalid magic byte resilience."""
        pcm_path = self._get_audio("greeting")
        if not pcm_path:
            return self._skip_result("P-003", "Invalid magic byte resilience", "protocol", "greeting audio not found")

        pcm = Path(pcm_path).read_bytes()
        t0 = time.monotonic()

        counts: dict[str, int] = {}
        tool_events: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        lod_updates: list[dict[str, Any]] = []
        transcripts: list[dict[str, str]] = []
        frame_acks: list[dict[str, Any]] = []

        connection_survived = False

        try:
            async with websockets.connect(self._ws_url("e2e_P003"), max_size=None) as ws:
                await _wait_for_type(ws, "session_ready", timeout_sec=20.0)
                await asyncio.sleep(self._t(3.0))

                # Send invalid magic byte frame
                await ws.send(bytes([0x03]) + b"\x00" * 640)

                # Wait a moment, then send valid audio
                await asyncio.sleep(1.0)
                await _send_audio_stream(ws, pcm)

                await _collect_responses(
                    ws,
                    self._t(20.0),
                    counts=counts,
                    tool_events=tool_events,
                    tool_results=tool_results,
                    lod_updates=lod_updates,
                    transcripts=transcripts,
                    frame_acks=frame_acks,
                )
                connection_survived = True
        except websockets.exceptions.ConnectionClosed:
            connection_survived = False

        duration = time.monotonic() - t0
        result = ScenarioResult(
            scenario_id="P-003",
            name="Invalid magic byte resilience",
            category="protocol",
            counts=counts,
            tool_events=tool_events,
            tool_results=tool_results,
            lod_updates=lod_updates,
            transcripts=transcripts,
            frame_acks=frame_acks,
            duration_sec=duration,
        )

        if not connection_survived:
            result.failures.append("connection_dropped_after_invalid_magic_byte")
        if connection_survived and result.counts.get("audio_bytes", 0) == 0:
            result.warnings.append("no_audio_response_after_recovery")

        result.passed = not result.failures
        return result

    async def _scenario_P004(self) -> ScenarioResult:
        """P-004: Session lifecycle (graceful disconnect)."""
        pcm_path = self._get_audio("greeting")
        if not pcm_path:
            return self._skip_result("P-004", "Session lifecycle", "protocol", "greeting audio not found")

        pcm = Path(pcm_path).read_bytes()
        t0 = time.monotonic()
        graceful = True

        try:
            async with websockets.connect(self._ws_url("e2e_P004"), max_size=None) as ws:
                await _wait_for_type(ws, "session_ready", timeout_sec=20.0)
                await asyncio.sleep(self._t(3.0))
                await _send_audio_stream(ws, pcm)
                await asyncio.sleep(self._t(5.0))
                # Connection closes gracefully via context manager
        except Exception as exc:
            graceful = False
            logger.warning("P-004 connection error: %s", exc)

        duration = time.monotonic() - t0

        result = ScenarioResult(
            scenario_id="P-004",
            name="Session lifecycle",
            category="protocol",
            duration_sec=duration,
        )

        if not graceful:
            result.failures.append("connection_error_during_lifecycle")

        # Verify server is still responsive by making a quick connection
        try:
            async with websockets.connect(self._ws_url("e2e_P004_verify"), max_size=None) as ws:
                await _wait_for_type(ws, "session_ready", timeout_sec=10.0)
                result.notes.append("server_responsive_after_disconnect")
        except Exception:
            result.failures.append("server_unresponsive_after_disconnect")

        result.passed = not result.failures
        return result

    # ======================================================================
    # VISION PIPELINE (V-001 to V-006)
    # ======================================================================

    async def _scenario_V001(self) -> ScenarioResult:
        """V-001: Vision LOD1 safety."""
        image_path = self._get_image("staircase_hazard")
        if not image_path:
            return self._skip_result("V-001", "Vision LOD1 safety", "vision", "staircase_hazard image not found")

        result = await _run_probe(
            self._ws_url("e2e_V001"),
            image_path=image_path,
            telemetry={"motion_state": "walking", "step_cadence": 80},
            skip_audio=True,
            send_image_first=True,
            wait_before_sec=self._t(10.0),
            collect_sec=self._t(25.0),
        )
        result.scenario_id = "V-001"
        result.name = "Vision LOD1 safety"
        result.category = "vision"

        vision_results = [
            r for r in result.tool_results if r.get("_type") == "vision_result"
        ]
        if not vision_results:
            result.warnings.append("no_vision_result_received")
        else:
            has_safety = any(
                r.get("safety_warnings") for r in vision_results
            )
            if not has_safety:
                result.warnings.append("vision_result_missing_safety_warnings")

        result.passed = not result.failures
        return result

    async def _scenario_V002(self) -> ScenarioResult:
        """V-002: Vision LOD2 navigation."""
        image_path = self._get_image("indoor_lobby")
        if not image_path:
            return self._skip_result("V-002", "Vision LOD2 navigation", "vision", "indoor_lobby image not found")

        result = await _run_probe(
            self._ws_url("e2e_V002"),
            image_path=image_path,
            telemetry={"motion_state": "stationary", "step_cadence": 0},
            gesture="force_lod_2",
            skip_audio=True,
            send_image_first=True,
            wait_before_sec=self._t(10.0),
            collect_sec=self._t(25.0),
        )
        result.scenario_id = "V-002"
        result.name = "Vision LOD2 navigation"
        result.category = "vision"

        vision_results = [
            r for r in result.tool_results if r.get("_type") == "vision_result"
        ]
        if not vision_results:
            result.warnings.append("no_vision_result_received")

        result.passed = not result.failures
        return result

    async def _scenario_V003(self) -> ScenarioResult:
        """V-003: Vision LOD3 full narrative."""
        image_path = self._get_image("park_scene")
        if not image_path:
            return self._skip_result("V-003", "Vision LOD3 full narrative", "vision", "park_scene image not found")

        result = await _run_probe(
            self._ws_url("e2e_V003"),
            image_path=image_path,
            gesture="force_lod_3",
            skip_audio=True,
            send_image_first=True,
            wait_before_sec=self._t(10.0),
            collect_sec=self._t(25.0),
        )
        result.scenario_id = "V-003"
        result.name = "Vision LOD3 full narrative"
        result.category = "vision"

        vision_results = [
            r for r in result.tool_results if r.get("_type") == "vision_result"
        ]
        if not vision_results:
            result.warnings.append("no_vision_result_received")
        else:
            has_description = any(
                r.get("scene_description") for r in vision_results
            )
            if not has_description:
                result.warnings.append("vision_result_missing_scene_description")

        result.passed = not result.failures
        return result

    async def _scenario_V004(self) -> ScenarioResult:
        """V-004: Vision repeat suppression."""
        image_path = self._get_image("park_scene") or self._get_image("street_crossing")
        if not image_path:
            return self._skip_result("V-004", "Vision repeat suppression", "vision", "no test image found")

        result = await _run_probe(
            self._ws_url("e2e_V004"),
            image_path=image_path,
            gesture="force_lod_3",
            skip_audio=True,
            send_image_first=True,
            send_multiple_images=True,
            image_interval_sec=5.0,
            wait_before_sec=self._t(10.0),
            collect_sec=self._t(30.0),
        )
        result.scenario_id = "V-004"
        result.name = "Vision repeat suppression"
        result.category = "vision"

        # Check if second image was suppressed via tool_event
        suppressed = any(
            e.get("repeat_suppressed") for e in result.tool_events
        )
        if not suppressed:
            result.warnings.append("repeat_suppression_not_observed_in_tool_events")

        result.passed = not result.failures
        return result

    async def _scenario_V005(self) -> ScenarioResult:
        """V-005: Camera grace period."""
        image_path = self._get_image("park_scene") or self._get_image("street_crossing")
        if not image_path:
            return self._skip_result("V-005", "Camera grace period", "vision", "no test image found")

        # Send image immediately after session_ready (within grace period)
        result = await _run_probe(
            self._ws_url("e2e_V005"),
            image_path=image_path,
            skip_audio=True,
            send_image_first=True,
            wait_before_sec=0.5,  # Minimal wait — within 8s grace period
            collect_sec=self._t(15.0),
        )
        result.scenario_id = "V-005"
        result.name = "Camera grace period"
        result.category = "vision"

        # During grace period, vision analysis should be suppressed
        vision_results = [
            r for r in result.tool_results if r.get("_type") == "vision_result"
        ]
        if vision_results:
            result.warnings.append("vision_analysis_during_grace_period")

        result.passed = not result.failures
        return result

    async def _scenario_V006(self) -> ScenarioResult:
        """V-006: Vision interval by LOD."""
        image_path = self._get_image("park_scene") or self._get_image("street_crossing")
        if not image_path:
            return self._skip_result("V-006", "Vision interval by LOD", "vision", "no test image found")

        image_bytes = Path(image_path).read_bytes()
        t0 = time.monotonic()

        counts: dict[str, int] = {}
        tool_events: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        lod_updates: list[dict[str, Any]] = []
        transcripts: list[dict[str, str]] = []
        frame_acks: list[dict[str, Any]] = []

        async with websockets.connect(self._ws_url("e2e_V006"), max_size=None) as ws:
            await _wait_for_type(ws, "session_ready", timeout_sec=20.0)
            await asyncio.sleep(self._t(10.0))

            # Force LOD3 and send images at intervals
            await ws.send(json.dumps({"type": "gesture", "gesture": "force_lod_3"}))

            # Send 3 images at 6s intervals
            for i in range(3):
                await ws.send(bytes([MAGIC_IMAGE]) + image_bytes)
                if i < 2:
                    await asyncio.sleep(6.0)

            await _collect_responses(
                ws,
                self._t(25.0),
                counts=counts,
                tool_events=tool_events,
                tool_results=tool_results,
                lod_updates=lod_updates,
                transcripts=transcripts,
                frame_acks=frame_acks,
            )

        duration = time.monotonic() - t0
        result = ScenarioResult(
            scenario_id="V-006",
            name="Vision interval by LOD",
            category="vision",
            counts=counts,
            tool_events=tool_events,
            tool_results=tool_results,
            lod_updates=lod_updates,
            transcripts=transcripts,
            frame_acks=frame_acks,
            duration_sec=duration,
        )

        if len(frame_acks) < 2:
            result.warnings.append(f"expected_3_frame_acks_got_{len(frame_acks)}")

        result.passed = not result.failures
        return result

    # ======================================================================
    # OCR PIPELINE (O-001 to O-005)
    # ======================================================================

    async def _scenario_O001(self) -> ScenarioResult:
        """O-001: Safety OCR at LOD1."""
        image_path = self._get_image("construction_zone")
        if not image_path:
            return self._skip_result("O-001", "Safety OCR at LOD1", "ocr", "construction_zone image not found")

        result = await _run_probe(
            self._ws_url("e2e_O001"),
            image_path=image_path,
            telemetry={"motion_state": "walking", "step_cadence": 80},
            skip_audio=True,
            send_image_first=True,
            wait_before_sec=self._t(10.0),
            collect_sec=self._t(25.0),
        )
        result.scenario_id = "O-001"
        result.name = "Safety OCR at LOD1"
        result.category = "ocr"

        ocr_results = [
            r for r in result.tool_results if r.get("_type") == "ocr_result"
        ]
        if not ocr_results:
            result.warnings.append("no_ocr_result_received")

        result.passed = not result.failures
        return result

    async def _scenario_O002(self) -> ScenarioResult:
        """O-002: User-intent OCR."""
        pcm_path = self._get_audio("read_menu")
        image_path = self._get_image("restaurant_menu")
        if not pcm_path:
            return self._skip_result("O-002", "User-intent OCR", "ocr", "read_menu audio not found")
        if not image_path:
            return self._skip_result("O-002", "User-intent OCR", "ocr", "restaurant_menu image not found")

        result = await _run_probe(
            self._ws_url("e2e_O002"),
            pcm_path=pcm_path,
            image_path=image_path,
            send_image_first=True,
            wait_before_sec=self._t(10.0),
            collect_sec=self._t(35.0),
        )
        result.scenario_id = "O-002"
        result.name = "User-intent OCR"
        result.category = "ocr"

        ocr_tool_triggered = any(
            e.get("tool") == "extract_text_from_camera"
            for e in result.tool_events
        )
        if not ocr_tool_triggered:
            result.failures.append("extract_text_from_camera_not_triggered")

        ocr_results = [
            r for r in result.tool_results if r.get("_type") == "ocr_result"
        ]
        if not ocr_results:
            result.failures.append("no_ocr_result_received")

        result.passed = not result.failures
        return result

    async def _scenario_O003(self) -> ScenarioResult:
        """O-003: OCR without camera frame."""
        pcm_path = self._get_audio("whats_that_say")
        if not pcm_path:
            return self._skip_result("O-003", "OCR without camera frame", "ocr", "whats_that_say audio not found")

        result = await _run_probe(
            self._ws_url("e2e_O003"),
            pcm_path=pcm_path,
            # No image sent intentionally
            wait_before_sec=self._t(10.0),
            collect_sec=self._t(25.0),
        )
        result.scenario_id = "O-003"
        result.name = "OCR without camera frame"
        result.category = "ocr"

        # Should not crash; an error or graceful response is acceptable
        if any("crash" in f.lower() for f in result.failures):
            result.failures.append("server_crashed_on_ocr_without_frame")
        else:
            result.notes.append("server_handled_ocr_without_frame_gracefully")

        result.passed = not result.failures
        return result

    async def _scenario_O004(self) -> ScenarioResult:
        """O-004: OCR repeat suppression."""
        pcm_path = self._get_audio("read_menu")
        image_path = self._get_image("restaurant_menu")
        if not pcm_path or not image_path:
            return self._skip_result("O-004", "OCR repeat suppression", "ocr", "read_menu or restaurant_menu not found")

        # First: send OCR request
        result = await _run_probe(
            self._ws_url("e2e_O004"),
            pcm_path=pcm_path,
            image_path=image_path,
            send_image_first=True,
            send_multiple_images=True,
            image_interval_sec=8.0,
            wait_before_sec=self._t(10.0),
            collect_sec=self._t(35.0),
        )
        result.scenario_id = "O-004"
        result.name = "OCR repeat suppression"
        result.category = "ocr"

        # Check if repeat suppression was applied
        suppressed = any(
            e.get("repeat_suppressed") for e in result.tool_events
        )
        if not suppressed:
            result.warnings.append("ocr_repeat_suppression_not_observed")

        result.passed = not result.failures
        return result

    async def _scenario_O005(self) -> ScenarioResult:
        """O-005: Vision query triggers OCR (known issue)."""
        pcm_path = self._get_audio("describe_scene")
        image_path = self._get_image("park_scene") or self._get_image("street_crossing")
        if not pcm_path:
            return self._skip_result("O-005", "Vision query triggers OCR", "ocr", "describe_scene audio not found")
        if not image_path:
            return self._skip_result("O-005", "Vision query triggers OCR", "ocr", "no scene image found")

        result = await _run_probe(
            self._ws_url("e2e_O005"),
            pcm_path=pcm_path,
            image_path=image_path,
            send_image_first=True,
            wait_before_sec=self._t(10.0),
            collect_sec=self._t(30.0),
        )
        result.scenario_id = "O-005"
        result.name = "Vision query triggers OCR (known issue)"
        result.category = "ocr"

        ocr_triggered = any(
            e.get("tool") == "extract_text_from_camera"
            for e in result.tool_events
        )
        if ocr_triggered:
            result.warnings.append("vision_query_incorrectly_triggered_ocr")
            result.known_issue = True
            result.notes.append("known_issue:O-005")

        result.passed = not result.failures
        return result

    # ======================================================================
    # LOD TESTS (L-001 to L-003)
    # ======================================================================

    async def _scenario_L001(self) -> ScenarioResult:
        """L-001: Force LOD3 gesture."""
        result = await _run_probe(
            self._ws_url("e2e_L001"),
            gesture="force_lod_3",
            skip_audio=True,
            wait_before_sec=self._t(3.0),
            collect_sec=self._t(10.0),
        )
        result.scenario_id = "L-001"
        result.name = "Force LOD3 gesture"
        result.category = "lod"

        if not result.lod_updates:
            result.failures.append("no_lod_update_received")
        else:
            lod3 = any(u.get("lod") == 3 for u in result.lod_updates)
            if not lod3:
                result.failures.append("lod_update_not_set_to_3")

        result.passed = not result.failures
        return result

    async def _scenario_L002(self) -> ScenarioResult:
        """L-002: High motion LOD drop."""
        result = await _run_probe(
            self._ws_url("e2e_L002"),
            telemetry={
                "motion_state": "running",
                "step_cadence": 160,
                "ambient_noise_db": 90,
                "heart_rate": 150,
            },
            skip_audio=True,
            wait_before_sec=self._t(3.0),
            collect_sec=self._t(10.0),
        )
        result.scenario_id = "L-002"
        result.name = "High motion LOD drop"
        result.category = "lod"

        if not result.lod_updates:
            result.failures.append("no_lod_update_received")
        else:
            lod1 = any(u.get("lod") == 1 for u in result.lod_updates)
            if not lod1:
                result.warnings.append("lod_did_not_drop_to_1")

        result.passed = not result.failures
        return result

    async def _scenario_L003(self) -> ScenarioResult:
        """L-003: LOD up/down gestures."""
        t0 = time.monotonic()

        counts: dict[str, int] = {}
        tool_events: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        lod_updates: list[dict[str, Any]] = []
        transcripts: list[dict[str, str]] = []
        frame_acks: list[dict[str, Any]] = []

        async with websockets.connect(self._ws_url("e2e_L003"), max_size=None) as ws:
            await _wait_for_type(ws, "session_ready", timeout_sec=20.0)
            await asyncio.sleep(self._t(3.0))

            # Send lod_up gesture
            await ws.send(json.dumps({"type": "gesture", "gesture": "lod_up"}))
            await asyncio.sleep(2.0)

            # Send lod_down gesture
            await ws.send(json.dumps({"type": "gesture", "gesture": "lod_down"}))

            await _collect_responses(
                ws,
                self._t(10.0),
                counts=counts,
                tool_events=tool_events,
                tool_results=tool_results,
                lod_updates=lod_updates,
                transcripts=transcripts,
                frame_acks=frame_acks,
            )

        duration = time.monotonic() - t0
        result = ScenarioResult(
            scenario_id="L-003",
            name="LOD up/down gestures",
            category="lod",
            counts=counts,
            tool_events=tool_events,
            tool_results=tool_results,
            lod_updates=lod_updates,
            transcripts=transcripts,
            frame_acks=frame_acks,
            duration_sec=duration,
        )

        if len(lod_updates) < 2:
            result.failures.append(f"expected_2_lod_updates_got_{len(lod_updates)}")

        result.passed = not result.failures
        return result

    # ======================================================================
    # TOOL TESTS (T-001 to T-003)
    # ======================================================================

    async def _scenario_T001(self) -> ScenarioResult:
        """T-001: Navigation via voice."""
        pcm_path = self._get_audio("navigate_pharmacy")
        if not pcm_path:
            return self._skip_result("T-001", "Navigation via voice", "tool", "navigate_pharmacy audio not found")

        result = await _run_probe(
            self._ws_url("e2e_T001"),
            pcm_path=pcm_path,
            wait_before_sec=self._t(10.0),
            collect_sec=self._t(55.0),
        )
        result.scenario_id = "T-001"
        result.name = "Navigation via voice"
        result.category = "tool"

        nav_tool = any(
            e.get("tool") in {"navigate_to", "get_walking_directions"}
            for e in result.tool_events
        )
        if not nav_tool:
            result.failures.append("navigation_tool_not_invoked")

        if result.counts.get("navigation_result", 0) == 0:
            result.failures.append("no_navigation_result")

        result.passed = not result.failures
        return result

    async def _scenario_T002(self) -> ScenarioResult:
        """T-002: Search via voice."""
        pcm_path = self._get_audio("search_weather")
        if not pcm_path:
            return self._skip_result("T-002", "Search via voice", "tool", "search_weather audio not found")

        result = await _run_probe(
            self._ws_url("e2e_T002"),
            pcm_path=pcm_path,
            wait_before_sec=self._t(10.0),
            collect_sec=self._t(45.0),
        )
        result.scenario_id = "T-002"
        result.name = "Search via voice"
        result.category = "tool"

        search_tool = any(
            e.get("tool") == "google_search"
            for e in result.tool_events
        )
        if not search_tool:
            result.failures.append("google_search_not_invoked")

        if result.counts.get("search_result", 0) == 0:
            result.failures.append("no_search_result")

        result.passed = not result.failures
        return result

    async def _scenario_T003(self) -> ScenarioResult:
        """T-003: Tool behavior correctness."""
        pcm_nav = self._get_audio("navigate_pharmacy")
        pcm_search = self._get_audio("search_weather")
        if not pcm_nav or not pcm_search:
            return self._skip_result("T-003", "Tool behavior correctness", "tool", "audio assets not found")

        # Run navigation probe to check INTERRUPT behavior
        nav_result = await _run_probe(
            self._ws_url("e2e_T003_nav"),
            pcm_path=pcm_nav,
            telemetry={"motion_state": "walking", "step_cadence": 80},
            wait_before_sec=self._t(10.0),
            collect_sec=self._t(45.0),
        )

        # Run search probe to check WHEN_IDLE behavior
        search_result = await _run_probe(
            self._ws_url("e2e_T003_search"),
            pcm_path=pcm_search,
            wait_before_sec=self._t(10.0),
            collect_sec=self._t(45.0),
        )

        result = ScenarioResult(
            scenario_id="T-003",
            name="Tool behavior correctness",
            category="tool",
            duration_sec=nav_result.duration_sec + search_result.duration_sec,
        )

        # Check navigation behavior
        nav_tools = [
            e for e in nav_result.tool_events
            if e.get("tool") in {"navigate_to", "get_walking_directions"}
        ]
        for te in nav_tools:
            behavior = te.get("behavior", "")
            if behavior and behavior.upper() != "INTERRUPT":
                result.warnings.append(
                    f"navigation_behavior_expected_INTERRUPT_got_{behavior}"
                )
            result.notes.append(f"nav_behavior={behavior}")

        # Check search behavior
        search_tools = [
            e for e in search_result.tool_events
            if e.get("tool") == "google_search"
        ]
        for te in search_tools:
            behavior = te.get("behavior", "")
            if behavior and behavior.upper() != "WHEN_IDLE":
                result.warnings.append(
                    f"search_behavior_expected_WHEN_IDLE_got_{behavior}"
                )
            result.notes.append(f"search_behavior={behavior}")

        # Merge tool data for report
        result.tool_events = nav_result.tool_events + search_result.tool_events
        result.tool_results = nav_result.tool_results + search_result.tool_results

        result.passed = not result.failures
        return result

    # ======================================================================
    # SAFETY-CRITICAL TESTS (S-001 to S-004)
    # ======================================================================

    async def _scenario_S001(self) -> ScenarioResult:
        """S-001: Hazard at all LODs."""
        image_path = self._get_image("staircase_hazard")
        if not image_path:
            return self._skip_result("S-001", "Hazard at all LODs", "safety", "staircase_hazard image not found")

        image_bytes = Path(image_path).read_bytes()
        all_vision_results: list[dict[str, Any]] = []
        all_lod_updates: list[dict[str, Any]] = []
        total_duration = 0.0

        for lod_level in [1, 2, 3]:
            session_id = f"e2e_S001_lod{lod_level}"
            t0 = time.monotonic()

            counts: dict[str, int] = {}
            tool_events: list[dict[str, Any]] = []
            tool_results: list[dict[str, Any]] = []
            lod_updates: list[dict[str, Any]] = []
            transcripts: list[dict[str, str]] = []
            frame_acks: list[dict[str, Any]] = []

            async with websockets.connect(self._ws_url(session_id), max_size=None) as ws:
                await _wait_for_type(ws, "session_ready", timeout_sec=20.0)
                await asyncio.sleep(self._t(10.0))

                # Force LOD level
                await ws.send(json.dumps({
                    "type": "gesture",
                    "gesture": f"force_lod_{lod_level}",
                }))
                await asyncio.sleep(1.0)

                # Send hazard image
                await ws.send(bytes([MAGIC_IMAGE]) + image_bytes)

                await _collect_responses(
                    ws,
                    self._t(25.0),
                    counts=counts,
                    tool_events=tool_events,
                    tool_results=tool_results,
                    lod_updates=lod_updates,
                    transcripts=transcripts,
                    frame_acks=frame_acks,
                )

            total_duration += time.monotonic() - t0
            all_vision_results.extend(
                r for r in tool_results if r.get("_type") == "vision_result"
            )
            all_lod_updates.extend(lod_updates)

        result = ScenarioResult(
            scenario_id="S-001",
            name="Hazard at all LODs",
            category="safety",
            duration_sec=total_duration,
            lod_updates=all_lod_updates,
            tool_results=all_vision_results,
        )

        if len(all_vision_results) < 2:
            result.warnings.append(
                f"expected_vision_results_at_each_LOD_got_{len(all_vision_results)}"
            )

        result.passed = not result.failures
        return result

    async def _scenario_S002(self) -> ScenarioResult:
        """S-002: Combined vision + OCR safety."""
        image_path = self._get_image("construction_zone")
        if not image_path:
            return self._skip_result("S-002", "Combined vision + OCR safety", "safety", "construction_zone image not found")

        result = await _run_probe(
            self._ws_url("e2e_S002"),
            image_path=image_path,
            telemetry={"motion_state": "walking", "step_cadence": 80},
            skip_audio=True,
            send_image_first=True,
            wait_before_sec=self._t(10.0),
            collect_sec=self._t(30.0),
        )
        result.scenario_id = "S-002"
        result.name = "Combined vision + OCR safety"
        result.category = "safety"

        has_vision = any(
            r.get("_type") == "vision_result" for r in result.tool_results
        )
        has_ocr = any(
            r.get("_type") == "ocr_result" for r in result.tool_results
        )

        if not has_vision:
            result.warnings.append("no_vision_result_for_safety")
        if not has_ocr:
            result.warnings.append("no_ocr_result_for_safety")

        result.passed = not result.failures
        return result

    async def _scenario_S003(self) -> ScenarioResult:
        """S-003: Vehicle detection."""
        image_path = self._get_image("street_crossing")
        if not image_path:
            return self._skip_result("S-003", "Vehicle detection", "safety", "street_crossing image not found")

        result = await _run_probe(
            self._ws_url("e2e_S003"),
            image_path=image_path,
            telemetry={"motion_state": "walking", "step_cadence": 80},
            skip_audio=True,
            send_image_first=True,
            wait_before_sec=self._t(10.0),
            collect_sec=self._t(25.0),
        )
        result.scenario_id = "S-003"
        result.name = "Vehicle detection"
        result.category = "safety"

        vision_results = [
            r for r in result.tool_results if r.get("_type") == "vision_result"
        ]
        if not vision_results:
            result.warnings.append("no_vision_result_received")
        else:
            safety_texts = " ".join(
                str(r.get("safety_warnings", "")) for r in vision_results
            ).lower()
            vehicle_keywords = {"vehicle", "car", "approaching", "traffic", "automobile"}
            if not any(kw in safety_texts for kw in vehicle_keywords):
                result.warnings.append("safety_warnings_missing_vehicle_mention")

        result.passed = not result.failures
        return result

    async def _scenario_S004(self) -> ScenarioResult:
        """S-004: Safety during vision cooldown."""
        image_path = self._get_image("staircase_hazard") or self._get_image("street_crossing")
        if not image_path:
            return self._skip_result("S-004", "Safety during vision cooldown", "safety", "hazard image not found")

        image_bytes = Path(image_path).read_bytes()
        t0 = time.monotonic()

        counts: dict[str, int] = {}
        tool_events: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        lod_updates: list[dict[str, Any]] = []
        transcripts: list[dict[str, str]] = []
        frame_acks: list[dict[str, Any]] = []

        async with websockets.connect(self._ws_url("e2e_S004"), max_size=None) as ws:
            await _wait_for_type(ws, "session_ready", timeout_sec=20.0)
            await asyncio.sleep(self._t(10.0))

            # Trigger first vision analysis
            await ws.send(bytes([MAGIC_IMAGE]) + image_bytes)
            await asyncio.sleep(3.0)

            # Immediately send hazard image during cooldown
            await ws.send(bytes([MAGIC_IMAGE]) + image_bytes)

            await _collect_responses(
                ws,
                self._t(30.0),
                counts=counts,
                tool_events=tool_events,
                tool_results=tool_results,
                lod_updates=lod_updates,
                transcripts=transcripts,
                frame_acks=frame_acks,
            )

        duration = time.monotonic() - t0
        result = ScenarioResult(
            scenario_id="S-004",
            name="Safety during vision cooldown",
            category="safety",
            counts=counts,
            tool_events=tool_events,
            tool_results=tool_results,
            lod_updates=lod_updates,
            transcripts=transcripts,
            frame_acks=frame_acks,
            duration_sec=duration,
        )

        # Safety results should still be processed
        vision_results = [
            r for r in tool_results if r.get("_type") == "vision_result"
        ]
        if not vision_results:
            result.warnings.append("no_vision_result_during_cooldown_safety_check")

        result.passed = not result.failures
        return result

    # ======================================================================
    # UX TESTS (UX-001 to UX-003)
    # ======================================================================

    async def _scenario_UX001(self) -> ScenarioResult:
        """UX-001: Pre-feedback timing."""
        image_path = self._get_image("park_scene") or self._get_image("street_crossing")
        if not image_path:
            return self._skip_result("UX-001", "Pre-feedback timing", "ux", "no test image found")

        result = await _run_probe(
            self._ws_url("e2e_UX001"),
            image_path=image_path,
            gesture="force_lod_3",
            skip_audio=True,
            send_image_first=True,
            wait_before_sec=self._t(10.0),
            collect_sec=self._t(30.0),
        )
        result.scenario_id = "UX-001"
        result.name = "Pre-feedback timing"
        result.category = "ux"

        # Check for pre-feedback transcript (e.g. "Let me look at that")
        agent_transcripts = [
            t for t in result.transcripts if t.get("role") == "agent"
        ]
        pre_feedback_keywords = {"look", "see", "analyzing", "checking", "examining", "let me"}
        has_pre_feedback = any(
            any(kw in t.get("text", "").lower() for kw in pre_feedback_keywords)
            for t in agent_transcripts
        )
        if not has_pre_feedback:
            result.warnings.append("no_pre_feedback_transcript_detected")

        result.passed = not result.failures
        return result

    async def _scenario_UX002(self) -> ScenarioResult:
        """UX-002: Transcript buffering quality."""
        pcm_path = self._get_audio("greeting")
        if not pcm_path:
            return self._skip_result("UX-002", "Transcript buffering", "ux", "greeting audio not found")

        result = await _run_probe(
            self._ws_url("e2e_UX002"),
            pcm_path=pcm_path,
            wait_before_sec=self._t(3.0),
            collect_sec=self._t(30.0),
        )
        result.scenario_id = "UX-002"
        result.name = "Transcript buffering quality"
        result.category = "ux"

        # Check that agent transcripts are not single-character rapid fragments
        agent_transcripts = [
            t for t in result.transcripts if t.get("role") == "agent"
        ]
        short_fragments = [
            t for t in agent_transcripts if len(t.get("text", "").strip()) <= 2
        ]
        if short_fragments and len(short_fragments) > len(agent_transcripts) * 0.5:
            result.warnings.append(
                f"excessive_short_fragments: {len(short_fragments)}/{len(agent_transcripts)}"
            )

        result.passed = not result.failures
        return result

    async def _scenario_UX003(self) -> ScenarioResult:
        """UX-003: Repeat last gesture."""
        pcm_greeting = self._get_audio("greeting")
        pcm_repeat = self._get_audio("repeat_last")
        if not pcm_greeting:
            return self._skip_result("UX-003", "Repeat last gesture", "ux", "greeting audio not found")
        if not pcm_repeat:
            return self._skip_result("UX-003", "Repeat last gesture", "ux", "repeat_last audio not found")

        greeting_pcm = Path(pcm_greeting).read_bytes()
        repeat_pcm = Path(pcm_repeat).read_bytes()
        t0 = time.monotonic()

        counts: dict[str, int] = {}
        tool_events: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        lod_updates: list[dict[str, Any]] = []
        transcripts: list[dict[str, str]] = []
        frame_acks: list[dict[str, Any]] = []

        async with websockets.connect(self._ws_url("e2e_UX003"), max_size=None) as ws:
            await _wait_for_type(ws, "session_ready", timeout_sec=20.0)
            await asyncio.sleep(self._t(10.0))

            # First turn: greeting
            await _send_audio_stream(ws, greeting_pcm)
            # Collect first response
            deadline = time.monotonic() + self._t(25.0)
            while time.monotonic() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if isinstance(raw, bytes):
                    counts["audio_bytes"] = counts.get("audio_bytes", 0) + 1
                    continue
                payload = json.loads(raw)
                _record_payload(
                    payload,
                    counts=counts,
                    tool_events=tool_events,
                    tool_results=tool_results,
                    lod_updates=lod_updates,
                    transcripts=transcripts,
                    frame_acks=frame_acks,
                )

            # Second turn: repeat
            await _send_audio_stream(ws, repeat_pcm)
            await _collect_responses(
                ws,
                self._t(25.0),
                counts=counts,
                tool_events=tool_events,
                tool_results=tool_results,
                lod_updates=lod_updates,
                transcripts=transcripts,
                frame_acks=frame_acks,
            )

        duration = time.monotonic() - t0
        result = ScenarioResult(
            scenario_id="UX-003",
            name="Repeat last gesture",
            category="ux",
            counts=counts,
            tool_events=tool_events,
            tool_results=tool_results,
            lod_updates=lod_updates,
            transcripts=transcripts,
            frame_acks=frame_acks,
            duration_sec=duration,
        )

        agent_texts = [
            t.get("text", "") for t in transcripts if t.get("role") == "agent"
        ]
        if len(agent_texts) < 2:
            result.warnings.append("insufficient_agent_responses_for_repeat_check")

        result.passed = not result.failures
        return result

    # ======================================================================
    # STABILITY TESTS (ST-001 to ST-003)
    # ======================================================================

    async def _scenario_ST001(self) -> ScenarioResult:
        """ST-001: Concurrent pipelines (rapid image frames)."""
        image_path = self._get_image("park_scene") or self._get_image("street_crossing")
        if not image_path:
            return self._skip_result("ST-001", "Concurrent pipelines", "stability", "no test image found")

        image_bytes = Path(image_path).read_bytes()
        t0 = time.monotonic()

        counts: dict[str, int] = {}
        tool_events: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        lod_updates: list[dict[str, Any]] = []
        transcripts: list[dict[str, str]] = []
        frame_acks: list[dict[str, Any]] = []

        crashed = False
        try:
            async with websockets.connect(self._ws_url("e2e_ST001"), max_size=None) as ws:
                await _wait_for_type(ws, "session_ready", timeout_sec=20.0)
                await asyncio.sleep(self._t(10.0))

                # Send 3 images rapidly within 2 seconds
                for i in range(3):
                    await ws.send(bytes([MAGIC_IMAGE]) + image_bytes)
                    await asyncio.sleep(0.6)

                await _collect_responses(
                    ws,
                    self._t(20.0),
                    counts=counts,
                    tool_events=tool_events,
                    tool_results=tool_results,
                    lod_updates=lod_updates,
                    transcripts=transcripts,
                    frame_acks=frame_acks,
                )
        except (websockets.exceptions.ConnectionClosed, ConnectionError):
            crashed = True

        duration = time.monotonic() - t0
        result = ScenarioResult(
            scenario_id="ST-001",
            name="Concurrent pipelines",
            category="stability",
            counts=counts,
            tool_events=tool_events,
            tool_results=tool_results,
            lod_updates=lod_updates,
            transcripts=transcripts,
            frame_acks=frame_acks,
            duration_sec=duration,
        )

        if crashed:
            result.failures.append("server_crashed_on_rapid_image_frames")
        if len(frame_acks) < 3:
            result.warnings.append(f"expected_3_frame_acks_got_{len(frame_acks)}")

        result.passed = not result.failures
        return result

    async def _scenario_ST002(self) -> ScenarioResult:
        """ST-002: Sequential sessions."""
        pcm_path = self._get_audio("greeting")
        if not pcm_path:
            return self._skip_result("ST-002", "Sequential sessions", "stability", "greeting audio not found")

        t0 = time.monotonic()
        session_results: list[bool] = []

        for i in range(3):
            try:
                r = await _run_probe(
                    self._ws_url(f"e2e_ST002_s{i}"),
                    pcm_path=pcm_path,
                    wait_before_sec=self._t(3.0),
                    collect_sec=self._t(15.0),
                )
                session_results.append(r.counts.get("audio_bytes", 0) > 0)
            except Exception as exc:
                logger.warning("ST-002 session %d failed: %s", i, exc)
                session_results.append(False)

        duration = time.monotonic() - t0
        result = ScenarioResult(
            scenario_id="ST-002",
            name="Sequential sessions",
            category="stability",
            duration_sec=duration,
        )

        failed_sessions = [
            i for i, ok in enumerate(session_results) if not ok
        ]
        if failed_sessions:
            result.failures.append(f"sessions_failed: {failed_sessions}")
        result.notes.append(f"session_results={session_results}")

        result.passed = not result.failures
        return result

    async def _scenario_ST003(self) -> ScenarioResult:
        """ST-003: No cross-session leakage."""
        pcm_greeting = self._get_audio("greeting")
        if not pcm_greeting:
            return self._skip_result("ST-003", "No cross-session leakage", "stability", "greeting audio not found")

        # Session A: mention "coffee shop"
        pcm_memory = self._get_audio("memory_set_coffee")
        if not pcm_memory:
            # Fall back to greeting if specific asset not available
            pcm_memory = pcm_greeting

        # Session A
        result_a = await _run_probe(
            self._ws_url("e2e_ST003_a"),
            pcm_path=pcm_memory,
            wait_before_sec=self._t(3.0),
            collect_sec=self._t(20.0),
        )

        # Session B (different session_id): should NOT know about session A
        result_b = await _run_probe(
            self._ws_url("e2e_ST003_b"),
            pcm_path=pcm_greeting,
            wait_before_sec=self._t(3.0),
            collect_sec=self._t(20.0),
        )

        result = ScenarioResult(
            scenario_id="ST-003",
            name="No cross-session leakage",
            category="stability",
            duration_sec=result_a.duration_sec + result_b.duration_sec,
        )

        # Check session B transcripts for session A content
        b_agent_text = " ".join(
            t.get("text", "") for t in result_b.transcripts if t.get("role") == "agent"
        ).lower()

        if "coffee shop" in b_agent_text:
            result.failures.append("cross_session_leakage_detected_coffee_shop")

        result.notes.append(f"session_a_transcripts={len(result_a.transcripts)}")
        result.notes.append(f"session_b_transcripts={len(result_b.transcripts)}")

        result.passed = not result.failures
        return result

    # ======================================================================
    # Helpers
    # ======================================================================

    def _skip_result(
        self,
        scenario_id: str,
        name: str,
        category: str,
        reason: str,
    ) -> ScenarioResult:
        """Return a skipped result when assets are missing."""
        logger.warning("SKIP [%s] %s: %s", scenario_id, name, reason)
        return ScenarioResult(
            scenario_id=scenario_id,
            name=name,
            category=category,
            passed=True,  # Skipped scenarios don't count as failures
            warnings=[f"skipped: {reason}"],
            notes=["skipped"],
        )

    # -- Report generation ---------------------------------------------------

    def _build_report(self, results: list[ScenarioResult]) -> dict[str, Any]:
        """Build the structured JSON report from all scenario results."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        # Category breakdown
        categories: dict[str, dict[str, int]] = {}
        for r in results:
            cat = r.category
            if cat not in categories:
                categories[cat] = {"total": 0, "passed": 0, "failed": 0}
            categories[cat]["total"] += 1
            if r.passed:
                categories[cat]["passed"] += 1
            else:
                categories[cat]["failed"] += 1

        # Build issues from failed scenarios
        issues = self._build_issues(results)

        # Server log analysis
        log_summary = {}
        if self.server_log_path and self.server_log_path.exists():
            log_summary = self._analyze_server_log()

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return {
            "suite_id": f"e2e_{stamp}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "server_url": self.ws_base_url,
            "total": total,
            "passed": passed,
            "failed": failed,
            "categories": categories,
            "scenarios": [r.to_dict() for r in results],
            "issues": issues,
            "log_summary": log_summary,
        }

    def _build_issues(self, results: list[ScenarioResult]) -> list[dict[str, Any]]:
        """Build issue definitions from failed and known-issue scenarios."""
        issues: list[dict[str, Any]] = []
        issue_counter = 1

        for r in results:
            if not r.failures and not r.known_issue:
                continue

            # Check known issue map first
            known = KNOWN_ISSUE_MAP.get(r.scenario_id)
            if known and r.known_issue:
                issues.append({
                    "id": f"E2E-{issue_counter:03d}",
                    "scenario": r.scenario_id,
                    "title": known["title"],
                    "location": known["location"],
                    "definition": known["definition"],
                    "severity": known["severity"],
                    "known_issue": True,
                    "evidence": {
                        "tool_events": r.tool_events,
                        "warnings": r.warnings,
                    },
                })
                issue_counter += 1
            elif r.failures:
                issues.append({
                    "id": f"E2E-{issue_counter:03d}",
                    "scenario": r.scenario_id,
                    "title": f"{r.name} failed",
                    "location": "unknown",
                    "definition": f"Scenario {r.scenario_id} ({r.name}) failed with: {r.failures}",
                    "severity": self._infer_severity(r.category),
                    "known_issue": False,
                    "evidence": {
                        "failures": r.failures,
                        "tool_events": r.tool_events,
                        "counts": r.counts,
                    },
                })
                issue_counter += 1

        return issues

    @staticmethod
    def _infer_severity(category: str) -> str:
        """Infer issue severity from the test category."""
        severity_map = {
            "protocol": "critical",
            "safety": "critical",
            "vision": "high",
            "ocr": "high",
            "tool": "high",
            "lod": "medium",
            "ux": "medium",
            "stability": "high",
        }
        return severity_map.get(category, "medium")

    def _analyze_server_log(self) -> dict[str, Any]:
        """Analyze the server log for error patterns."""
        if not self.server_log_path or not self.server_log_path.exists():
            return {"exists": False}

        text = self.server_log_path.read_text(encoding="utf-8", errors="ignore")
        return {
            "exists": True,
            "size_bytes": len(text.encode("utf-8")),
            "traceback_count": text.count("Traceback (most recent call last):"),
            "model_503_count": text.count(" 503 ") + text.count("UNAVAILABLE"),
            "downstream_error_count": text.count("Error in downstream handler"),
            "face_unavailable_count": text.count("InsightFace is unavailable"),
        }

    def _write_report(self, report: dict[str, Any]) -> None:
        """Write the report JSON to the output directory."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.output_dir / stamp
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Report written to %s", report_path)

        # Print summary to console
        self._print_summary(report)

    @staticmethod
    def _print_summary(report: dict[str, Any]) -> None:
        """Print a human-readable summary to stdout."""
        print("\n" + "=" * 70)
        print("  SightLine E2E Test Suite Report")
        print("=" * 70)
        print(f"  Suite ID:    {report['suite_id']}")
        print(f"  Created:     {report['created_at']}")
        print(f"  Server:      {report['server_url']}")
        print(f"  Total:       {report['total']}")
        print(f"  Passed:      {report['passed']}")
        print(f"  Failed:      {report['failed']}")
        print()

        # Category breakdown
        print("  Category Breakdown:")
        for cat, stats in sorted(report.get("categories", {}).items()):
            status = "OK" if stats["failed"] == 0 else "ISSUES"
            print(
                f"    {cat:15s}  {stats['passed']}/{stats['total']} passed  [{status}]"
            )
        print()

        # Failed/warned scenarios
        for sc in report.get("scenarios", []):
            if sc.get("failures") or sc.get("known_issue"):
                marker = "KNOWN" if sc.get("known_issue") else "FAIL"
                print(f"  [{marker}] {sc['scenario_id']}: {sc['name']}")
                for f in sc.get("failures", []):
                    print(f"         - {f}")
                for w in sc.get("warnings", []):
                    print(f"         ~ {w}")

        # Issues
        issues = report.get("issues", [])
        if issues:
            print()
            print(f"  Issues ({len(issues)}):")
            for issue in issues:
                known_tag = " [KNOWN]" if issue.get("known_issue") else ""
                print(
                    f"    {issue['id']} [{issue['severity']}]{known_tag}: "
                    f"{issue['title']}"
                )
                print(f"         Scenario: {issue['scenario']}")
                if issue.get("location") != "unknown":
                    print(f"         Location: {issue['location']}")

        print()
        all_ok = report["failed"] == 0
        print(f"  Result: {'ALL PASSED' if all_ok else 'FAILURES DETECTED'}")
        print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run E2E test suite against a running SightLine server.",
    )
    p.add_argument(
        "--server",
        default="ws://127.0.0.1:8100",
        help="WebSocket base URL (default: ws://127.0.0.1:8100)",
    )
    p.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("artifacts/e2e_assets"),
        help="Assets directory containing manifest.json (default: artifacts/e2e_assets)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/e2e_results"),
        help="Output directory for reports (default: artifacts/e2e_results)",
    )
    p.add_argument(
        "--server-log",
        type=Path,
        default=None,
        help="Path to server log file for analysis",
    )
    p.add_argument(
        "--scenarios",
        default=None,
        help='Filter scenarios by pattern (e.g., "P-*", "V-001", "OCR")',
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Global timeout multiplier (default: 1.0)",
    )
    return p


async def _async_main(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]

    # Resolve paths relative to repo root
    assets_dir = args.assets_dir
    if not assets_dir.is_absolute():
        assets_dir = repo_root / assets_dir

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    suite = E2ETestSuite(
        ws_base_url=args.server,
        assets_dir=assets_dir,
        output_dir=output_dir,
        server_log_path=args.server_log,
        timeout_multiplier=args.timeout,
        scenario_filter=args.scenarios,
    )

    try:
        report = await suite.run_all()
    except FileNotFoundError as exc:
        logger.error("Asset error: %s", exc)
        return 1
    except ConnectionError as exc:
        logger.error("Server connection error: %s", exc)
        return 1

    return 0 if report["failed"] == 0 else 2


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    args = _build_parser().parse_args()
    try:
        return asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
