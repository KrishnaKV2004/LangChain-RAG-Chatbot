"""FastAPI application: /chat, /upload, /reindex, /search, /health, /cache."""

from app.api.main import create_app

__all__ = ["create_app"]
