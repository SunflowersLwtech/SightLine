"""Tests for context.lod_evaluator module.

All LLM calls are mocked.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from context.lod_evaluator import (
    LODAdjustment,
    LODEvaluator,
    _DEBOUNCE_S,
    _TIMEOUT_S,
)
from lod.models import UserProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class FakeLocationCtx:
    def __init__(self, place_name="Cafe", place_type="cafe", familiarity_score=0.5):
        self.place_name = place_name
        self.place_type = place_type
        self.familiarity_score = familiarity_score


# ---------------------------------------------------------------------------
# LODAdjustment parsing
# ---------------------------------------------------------------------------


class TestParseResponse:
    def setup_method(self):
        self.evaluator = LODEvaluator()

    def test_parse_keep(self):
        result = self.evaluator._parse_response("DECISION: KEEP\nREASON: All good")
        assert result.delta == 0
        assert "ALL GOOD" in result.reason

    def test_parse_up(self):
        result = self.evaluator._parse_response("DECISION: UP\nREASON: New location")
        assert result.delta == 1

    def test_parse_down(self):
        result = self.evaluator._parse_response("DECISION: DOWN\nREASON: Familiar place")
        assert result.delta == -1

    def test_parse_malformed_defaults_keep(self):
        result = self.evaluator._parse_response("I think we should do nothing")
        assert result.delta == 0


# ---------------------------------------------------------------------------
# Safety floor
# ---------------------------------------------------------------------------


class TestSafetyFloor:
    def test_lod1_never_adjusted(self):
        evaluator = LODEvaluator()
        result = _run(evaluator.evaluate(baseline_lod=1))
        assert result.delta == 0
        assert "Safety" in result.reason

    def test_lod1_from_panic_never_adjusted(self):
        evaluator = LODEvaluator()
        result = _run(evaluator.evaluate(
            baseline_lod=1,
            location_ctx=FakeLocationCtx(familiarity_score=0.1),
        ))
        assert result.delta == 0


# ---------------------------------------------------------------------------
# Debounce
# ---------------------------------------------------------------------------


class TestDebounce:
    def test_rapid_calls_return_cached(self):
        evaluator = LODEvaluator()
        # Simulate a previous evaluation
        evaluator._last_eval_time = time.time()
        evaluator._last_result = LODAdjustment(delta=1, reason="cached", confidence=0.8)

        result = _run(evaluator.evaluate(baseline_lod=2))
        assert result.delta == 1
        assert result.reason == "cached"

    def test_debounce_no_previous_result(self):
        evaluator = LODEvaluator()
        evaluator._last_eval_time = time.time()
        evaluator._last_result = None

        result = _run(evaluator.evaluate(baseline_lod=2))
        assert result.delta == 0
        assert "Debounce" in result.reason


# ---------------------------------------------------------------------------
# LLM call mocking
# ---------------------------------------------------------------------------


class TestEvaluateWithMockedLLM:
    def test_evaluate_up(self):
        evaluator = LODEvaluator()

        async def mock_call_llm(prompt):
            return LODAdjustment(delta=1, reason="New location needs detail", confidence=0.8)

        evaluator._call_llm = mock_call_llm

        result = _run(evaluator.evaluate(
            baseline_lod=2,
            location_ctx=FakeLocationCtx(familiarity_score=0.1),
        ))
        assert result.delta == 1

    def test_evaluate_down(self):
        evaluator = LODEvaluator()

        async def mock_call_llm(prompt):
            return LODAdjustment(delta=-1, reason="Familiar location", confidence=0.8)

        evaluator._call_llm = mock_call_llm

        result = _run(evaluator.evaluate(
            baseline_lod=3,
            location_ctx=FakeLocationCtx(familiarity_score=0.9),
        ))
        assert result.delta == -1

    def test_evaluate_keep(self):
        evaluator = LODEvaluator()

        async def mock_call_llm(prompt):
            return LODAdjustment(delta=0, reason="All good", confidence=0.8)

        evaluator._call_llm = mock_call_llm

        result = _run(evaluator.evaluate(baseline_lod=2))
        assert result.delta == 0

    def test_timeout_returns_keep(self):
        evaluator = LODEvaluator()

        async def slow_call(prompt):
            await asyncio.sleep(2.0)  # Well over 500ms
            return LODAdjustment(delta=1, reason="Too late", confidence=0.8)

        evaluator._call_llm = slow_call

        result = _run(evaluator.evaluate(baseline_lod=2))
        assert result.delta == 0
        assert "Timeout" in result.reason

    def test_error_returns_keep(self):
        evaluator = LODEvaluator()

        async def failing_call(prompt):
            raise RuntimeError("API error")

        evaluator._call_llm = failing_call

        result = _run(evaluator.evaluate(baseline_lod=2))
        assert result.delta == 0
        assert "Error" in result.reason


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


class TestPromptBuilding:
    def test_prompt_includes_context(self):
        evaluator = LODEvaluator()
        prompt = evaluator._build_prompt(
            baseline_lod=2,
            location_ctx=FakeLocationCtx(place_name="Starbucks"),
            memories=[{"content": "User likes coffee"}],
            profile=UserProfile(om_level="advanced"),
            entities=None,
        )
        assert "Starbucks" in prompt
        assert "coffee" in prompt
        assert "advanced" in prompt

    def test_prompt_handles_none_context(self):
        evaluator = LODEvaluator()
        prompt = evaluator._build_prompt(
            baseline_lod=2,
            location_ctx=None,
            memories=None,
            profile=None,
            entities=None,
        )
        assert "unknown" in prompt
        assert "none" in prompt
