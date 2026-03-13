"""Compatibility regressions around ADK event schema and empty embeddings."""

from types import SimpleNamespace


def test_extract_function_calls_uses_event_getter():
    from dispatch.tool_dispatcher import _extract_function_calls

    call = SimpleNamespace(name="demo_tool", args={"x": 1})
    event = SimpleNamespace(
        get_function_calls=lambda: [call],
        actions=SimpleNamespace(),
    )

    calls = _extract_function_calls(event)

    assert len(calls) == 1
    assert calls[0].name == "demo_tool"


def test_extract_function_calls_handles_missing_actions_field_safely():
    from dispatch.tool_dispatcher import _extract_function_calls

    event = SimpleNamespace(
        get_function_calls=lambda: [],
        actions=SimpleNamespace(),  # No `function_calls` attribute in adk 1.25.x
    )

    assert _extract_function_calls(event) == []


def test_extract_function_calls_legacy_fallback():
    from dispatch.tool_dispatcher import _extract_function_calls

    call = SimpleNamespace(name="legacy_tool", args={})
    event = SimpleNamespace(
        actions=SimpleNamespace(function_calls=[call]),
    )

    calls = _extract_function_calls(event)

    assert len(calls) == 1
    assert calls[0].name == "legacy_tool"


def test_compute_embedding_empty_text_returns_none():
    from memory import memory_bank

    vec = memory_bank._compute_embedding("   ")

    assert vec is None
