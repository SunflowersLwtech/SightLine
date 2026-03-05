import asyncio
import json
from collections import deque
from pathlib import Path

import pytest

from scripts.run_multiturn_e2e import _run_single_turn


class _FakeWS:
    def __init__(self, incoming):
        self._incoming = deque(incoming)
        self.sent = []

    async def send(self, payload):
        self.sent.append(payload)

    async def recv(self):
        if self._incoming:
            return self._incoming.popleft()
        raise asyncio.TimeoutError()


@pytest.mark.asyncio
async def test_turn_validates_message_fields_and_behavior(tmp_path: Path):
    ws = _FakeWS(
        [
            json.dumps({"type": "transcript", "role": "user", "text": "hello"}),
            json.dumps({"type": "transcript", "role": "agent", "text": "I can help."}),
            b"\x00\x01",
            json.dumps(
                {
                    "type": "tool_event",
                    "tool": "navigate_to",
                    "status": "invoked",
                    "behavior": "INTERRUPT",
                }
            ),
            json.dumps(
                {
                    "type": "capability_degraded",
                    "capability": "camera",
                    "reason": "lens_blocked",
                }
            ),
        ]
    )
    pcm_cache = {"T01": b"\x00\x00" * 32}
    turn_def = {
        "id": "T01",
        "text": "please navigate now",
        "expect_agent_response": True,
        "expect_tool": "navigate_to",
        "expect_tool_behavior": {"navigate_to": "INTERRUPT"},
        "expect_message_types": ["capability_degraded"],
        "expect_message_fields": [
            {"type": "capability_degraded", "field": "capability", "equals": "camera"},
        ],
    }

    result = await _run_single_turn(
        ws=ws,
        turn_def=turn_def,
        pcm_cache=pcm_cache,
        image_dir=tmp_path,
        collect_sec=0.02,
        strict_expectations=True,
    )

    assert result.passed is True
    assert result.failures == []


@pytest.mark.asyncio
async def test_turn_non_strict_expectation_miss_becomes_warning(tmp_path: Path):
    ws = _FakeWS(
        [
            json.dumps({"type": "transcript", "role": "user", "text": "store this"}),
            json.dumps({"type": "transcript", "role": "agent", "text": "Sure."}),
            b"\x00\x01",
        ]
    )
    pcm_cache = {"T02": b"\x00\x00" * 24}
    turn_def = {
        "id": "T02",
        "text": "remember this",
        "expect_agent_response": True,
        "expect_tool": "remember_entity",
    }

    result = await _run_single_turn(
        ws=ws,
        turn_def=turn_def,
        pcm_cache=pcm_cache,
        image_dir=tmp_path,
        collect_sec=0.02,
        strict_expectations=False,
    )

    assert result.passed is True
    assert any("expected_tool_remember_entity_not_called" in w for w in result.warnings)


@pytest.mark.asyncio
async def test_turn_detects_json_leak_pattern(tmp_path: Path):
    ws = _FakeWS(
        [
            json.dumps({"type": "transcript", "role": "user", "text": "what happened"}),
            json.dumps({"type": "transcript", "role": "agent", "text": "{\"type\":\"telemetry\"}"}),
            b"\x00\x01",
        ]
    )
    pcm_cache = {"T03": b"\x00\x00" * 24}
    turn_def = {
        "id": "T03",
        "text": "status",
        "expect_agent_response": True,
    }

    result = await _run_single_turn(
        ws=ws,
        turn_def=turn_def,
        pcm_cache=pcm_cache,
        image_dir=tmp_path,
        collect_sec=0.02,
        strict_expectations=True,
    )

    assert result.passed is False
    assert any("context_regex_leaked" in f for f in result.failures)
