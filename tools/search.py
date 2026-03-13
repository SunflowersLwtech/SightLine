"""SightLine Google Search grounding tool.

Provides fact-checking and information retrieval via Gemini's built-in
Google Search grounding capability.  Used for verifying brand/product
identification, looking up business hours, menus, news, and events.

Behavior mode: WHEN_IDLE — the search is triggered when the model
finishes speaking, so results are delivered without interrupting
ongoing narration.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from google import genai
from google.genai import errors as genai_errors

logger = logging.getLogger("sightline.tools.search")

# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Return a lazily-initialised Genai client."""
    global _client
    if _client is None:
        api_key = os.environ.get("_GOOGLE_AI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable not set")
        _client = genai.Client(api_key=api_key, vertexai=False)
    return _client


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------

SEARCH_MODEL = "gemini-3-flash-preview"
_SEARCH_RETRY_ATTEMPTS = 3
_SEARCH_RETRY_BACKOFF_SEC = 0.4


def google_search(query: str) -> dict[str, Any]:
    """Search the web and return a summarised answer with sources.

    Uses Gemini's built-in Google Search grounding to get up-to-date
    information.  Suitable for:
    - Brand/product identification and verification
    - Business hours and contact info
    - Menu items and prices
    - News and events
    - General fact-checking

    Args:
        query: The search query or question to answer.

    Returns:
        Dict with ``answer``, ``sources``, and ``confidence``.
    """
    try:
        client = _get_client()
        response = _generate_search_with_retry(client=client, query=query)

        # Extract the answer text
        answer = ""
        if response.candidates and response.candidates[0].content:
            parts = response.candidates[0].content.parts
            if parts:
                answer = "".join(p.text for p in parts if p.text)

        # Extract grounding metadata / sources
        sources = _extract_sources(response)

        # Confidence based on whether we got grounding support
        confidence = 0.9 if sources else 0.5

        return {
            "success": True,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
        }

    except Exception as e:
        status_code = getattr(e, "status_code", None)
        err_text = str(e)
        is_unavailable = (
            isinstance(e, genai_errors.ServerError) and status_code == 503
        ) or ("UNAVAILABLE" in err_text.upper() and "503" in err_text)
        if is_unavailable:
            logger.warning("google_search unavailable (503/high demand): %s", err_text[:200])
            return {
                "success": False,
                "answer": "Search backend is temporarily unavailable. Please try again shortly.",
                "sources": [],
                "confidence": 0.0,
                "retryable": True,
                "error_code": "upstream_unavailable",
            }
        logger.exception("google_search failed: %s", e)
        return {
            "success": False,
            "answer": f"Search failed: {e}",
            "sources": [],
            "confidence": 0.0,
        }


def _generate_search_with_retry(*, client: genai.Client, query: str):
    last_exc: Exception | None = None
    for attempt in range(1, _SEARCH_RETRY_ATTEMPTS + 1):
        try:
            return client.models.generate_content(
                model=SEARCH_MODEL,
                contents=query,
                config=genai.types.GenerateContentConfig(
                    tools=[genai.types.Tool(google_search=genai.types.GoogleSearch())],
                ),
            )
        except Exception as exc:
            last_exc = exc
            if not _is_transient_search_error(exc) or attempt >= _SEARCH_RETRY_ATTEMPTS:
                raise
            backoff = _SEARCH_RETRY_BACKOFF_SEC * attempt
            logger.warning(
                "google_search transient failure (%s); retrying in %.2fs (%d/%d)",
                type(exc).__name__,
                backoff,
                attempt,
                _SEARCH_RETRY_ATTEMPTS,
            )
            time.sleep(backoff)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("google_search failed without an exception")


def _is_transient_search_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    err_text = str(exc).upper()
    if status_code in {429, 500, 502, 503, 504}:
        return True
    return any(
        token in err_text for token in (
            "RESOURCE_EXHAUSTED",
            "RATE LIMIT",
            "UNAVAILABLE",
            "DEADLINE_EXCEEDED",
            "503",
            "504",
            "429",
        )
    )


def _extract_sources(response: Any) -> list[dict[str, str]]:
    """Extract source information from a Gemini grounding response."""
    sources: list[dict[str, str]] = []

    if not response.candidates:
        return sources

    candidate = response.candidates[0]
    grounding_metadata = getattr(candidate, "grounding_metadata", None)
    if not grounding_metadata:
        return sources

    # Extract from grounding_chunks if available
    chunks = getattr(grounding_metadata, "grounding_chunks", None)
    if chunks:
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            if web:
                sources.append({
                    "title": getattr(web, "title", "") or "",
                    "url": getattr(web, "uri", "") or "",
                    "snippet": "",
                })

    # Extract from search_entry_point if available
    search_entry = getattr(grounding_metadata, "search_entry_point", None)
    if search_entry and not sources:
        rendered = getattr(search_entry, "rendered_content", "")
        if rendered:
            sources.append({
                "title": "Google Search",
                "url": "",
                "snippet": rendered[:200],
            })

    return sources


# ---------------------------------------------------------------------------
# ADK FunctionDeclaration for Gemini Live API
# ---------------------------------------------------------------------------

SEARCH_TOOL_DECLARATIONS = [
    {
        "name": "google_search",
        "description": (
            "Search the web for current information. Use for verifying brand or product "
            "names, looking up business hours, menus, prices, news, events, or any "
            "factual question that needs up-to-date information. Behavior: WHEN_IDLE — "
            "results are delivered after the model finishes speaking."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query or question to look up",
                },
            },
            "required": ["query"],
        },
    },
]

SEARCH_FUNCTIONS = {
    "google_search": google_search,
}
