"""HTTP client for the FastAPI backend.

The frontend never imports application code — it talks to the API over HTTP
exactly like any other client would, so the two processes can be deployed
and scaled independently.

Configuration (environment variables, also read from ``.env``):

* ``RAG_API_URL``   — backend base URL (default ``http://localhost:8000``)
* ``RAG_API_TOKEN`` — bearer token, required only when the API enforces auth.
  Falls back to ``API__AUTH_TOKEN`` so the backend's own setting is the single
  source of truth and a locally-run UI doesn't 401 the moment auth is enabled.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

# The UI runs as its own process, so nothing has loaded .env for it. Without
# this, enabling API__AUTH_TOKEN makes every request 401 until the operator
# happens to export the token by hand.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class APIError(Exception):
    """Raised when the backend returns an error or is unreachable."""


class RAGAPIClient:
    """Thin typed wrapper over the backend endpoints."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = (base_url or os.getenv("RAG_API_URL", "http://localhost:8000")).rstrip("/")
        self._token = token or os.getenv("RAG_API_TOKEN") or os.getenv("API__AUTH_TOKEN")
        self._timeout = timeout

    # ------------------------------------------------------------------ #

    def chat(
        self,
        query: str,
        history: List[Dict[str, str]],
        force_route: Optional[str] = None,
        role: str = "employee",
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"query": query, "history": history, "role": role}
        if force_route:
            payload["force_route"] = force_route
        return self._post("/chat", json=payload)

    def rates(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch structured freight quotes for an already-parsed rate query."""
        return self._post("/rates", json=query)

    def upload(self, filename: str, content: bytes) -> Dict[str, Any]:
        return self._request(
            "POST", "/upload", files={"file": (filename, content)}
        )

    def reindex(self, rebuild: bool = True) -> Dict[str, Any]:
        return self._post("/reindex", json={"rebuild": rebuild})

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def cache_stats(self) -> Dict[str, Any]:
        return self._request("GET", "/cache")

    def clear_cache(self, expired_only: bool = False) -> Dict[str, Any]:
        return self._request(
            "DELETE", "/cache", params={"expired_only": str(expired_only).lower()}
        )

    # ------------------------------------------------------------------ #

    def _post(self, path: str, json: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", path, json=json)

    def _request(self, method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
        headers = kwargs.pop("headers", {})
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        try:
            response = httpx.request(
                method,
                f"{self._base_url}{path}",
                headers=headers,
                timeout=self._timeout,
                **kwargs,
            )
        except httpx.HTTPError as exc:
            raise APIError(f"Backend unreachable at {self._base_url}: {exc}") from exc

        if response.status_code >= 400:
            try:
                detail = response.json().get("error") or response.json().get("detail")
            except Exception:  # noqa: BLE001 — non-JSON error body
                detail = response.text[:200]
            raise APIError(f"{response.status_code}: {detail}")
        return response.json()
