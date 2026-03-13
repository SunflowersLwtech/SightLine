from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from tools import _maps_http


def _mock_response(status_code: int, payload: dict | None = None) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = payload or {}
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "bad response",
            request=MagicMock(),
            response=response,
        )
    else:
        response.raise_for_status.return_value = None
    return response


def test_maps_rest_post_retries_transient_http_status():
    with patch("tools._maps_http._get_api_key", return_value="test-key"), \
         patch("tools._maps_http.httpx.request") as mock_request, \
         patch("tools._maps_http.time.sleep", return_value=None):
        mock_request.side_effect = [
            _mock_response(503),
            _mock_response(200, {"ok": True}),
        ]

        result = _maps_http.maps_rest_post("https://example.com", {"q": 1})

    assert result == {"ok": True}
    assert mock_request.call_count == 2


def test_maps_rest_get_retries_timeout_then_succeeds():
    with patch("tools._maps_http._get_api_key", return_value="test-key"), \
         patch("tools._maps_http.httpx.request") as mock_request, \
         patch("tools._maps_http.time.sleep", return_value=None):
        mock_request.side_effect = [
            httpx.TimeoutException("timeout"),
            _mock_response(200),
        ]

        resp = _maps_http.maps_rest_get("https://example.com")

    assert resp.status_code == 200
    assert mock_request.call_count == 2
