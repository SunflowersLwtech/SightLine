"""SightLine Gemini Maps Grounding tool.

Provides rich place information, reviews, and geographic reasoning via
Gemini's built-in Google Maps grounding.  This uses a standard Gemini
model (not Live API) as Maps grounding is not supported with Live API.

Uses Vertex AI mode (ADC authentication) since Maps grounding requires it.

Behavior mode: WHEN_IDLE — results delivered after speech finishes.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from google import genai
from google.genai import types

logger = logging.getLogger("sightline.tools.maps_grounding")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAPS_MODEL = "gemini-2.5-flash"  # Maps grounding supported model

# ---------------------------------------------------------------------------
# Client singleton (Vertex AI mode)
# ---------------------------------------------------------------------------

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Return a lazily-initialised Genai client in Vertex AI mode.

    Uses Application Default Credentials (ADC) for authentication.
    Different from search.py which uses API key mode.
    """
    global _client
    if _client is None:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "sightline-hackathon")
        location = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
        _client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
        )
    return _client


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------


def maps_query(
    question: str,
    lat: float = 0.0,
    lng: float = 0.0,
) -> dict[str, Any]:
    """Query Google Maps via Gemini's Maps grounding for place information.

    Suitable for open-ended geographic questions like:
    - "What good Chinese restaurants are nearby?"
    - "Is there a pharmacy open now?"
    - "What are the reviews for this place?"
    - "What's the closest accessible bathroom?"

    Args:
        question: The geographic/place question to answer.
        lat: User's latitude (0.0 if unknown).
        lng: User's longitude (0.0 if unknown).

    Returns:
        Dict with ``answer``, ``sources``, and ``confidence``.
    """
    try:
        client = _get_client()

        # Build the prompt with location context if available
        prompt_parts = []
        if lat != 0.0 or lng != 0.0:
            prompt_parts.append(
                f"The user is currently at coordinates ({lat:.6f}, {lng:.6f}). "
                "Consider this location when answering."
            )
        prompt_parts.append(question)
        prompt = " ".join(prompt_parts)

        response = client.models.generate_content(
            model=MAPS_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_maps=types.GoogleMaps())],
            ),
        )

        # Extract the answer text
        answer = ""
        if response.candidates and response.candidates[0].content:
            parts = response.candidates[0].content.parts
            if parts:
                answer = "".join(p.text for p in parts if p.text)

        # Extract Maps grounding sources
        sources = _extract_maps_sources(response)

        confidence = 0.9 if sources else 0.5

        return {
            "success": True,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
        }

    except Exception as e:
        logger.exception("maps_query failed: %s", e)
        return {
            "success": False,
            "answer": f"Maps query failed: {e}",
            "sources": [],
            "confidence": 0.0,
        }


def _extract_maps_sources(response: Any) -> list[dict[str, str]]:
    """Extract source information from a Gemini Maps grounding response."""
    sources: list[dict[str, str]] = []

    if not response.candidates:
        return sources

    candidate = response.candidates[0]
    grounding_metadata = getattr(candidate, "grounding_metadata", None)
    if not grounding_metadata:
        return sources

    # Extract from grounding_chunks (Maps results)
    chunks = getattr(grounding_metadata, "grounding_chunks", None)
    if chunks:
        for chunk in chunks:
            # Maps grounding may use 'web' or 'retrieved_context'
            web = getattr(chunk, "web", None)
            if web:
                sources.append({
                    "title": getattr(web, "title", "") or "",
                    "url": getattr(web, "uri", "") or "",
                })

    # Extract from grounding_supports for Maps-specific data
    supports = getattr(grounding_metadata, "grounding_supports", None)
    if supports and not sources:
        for support in supports:
            segment = getattr(support, "segment", None)
            if segment:
                text = getattr(segment, "text", "")
                if text:
                    sources.append({
                        "title": "Google Maps",
                        "snippet": text[:200],
                    })

    return sources


# ---------------------------------------------------------------------------
# ADK FunctionDeclaration for Gemini Live API
# ---------------------------------------------------------------------------

MAPS_GROUNDING_TOOL_DECLARATIONS = [
    {
        "name": "maps_query",
        "description": (
            "Query Google Maps for detailed place information, reviews, ratings, "
            "business hours, and geographic reasoning. Use for open-ended location "
            "questions like 'What good restaurants are nearby?', 'Is there an "
            "accessible pharmacy open now?', 'What are the reviews for this place?'. "
            "Different from nearby_search — this provides richer, more conversational "
            "answers with reviews and details. Behavior: WHEN_IDLE."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The geographic or place-related question to answer",
                },
                "lat": {
                    "type": "number",
                    "description": "User's latitude (auto-injected from GPS)",
                },
                "lng": {
                    "type": "number",
                    "description": "User's longitude (auto-injected from GPS)",
                },
            },
            "required": ["question"],
        },
    },
]

MAPS_GROUNDING_FUNCTIONS = {
    "maps_query": maps_query,
}
