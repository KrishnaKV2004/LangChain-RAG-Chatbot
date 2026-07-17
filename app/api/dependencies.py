"""FastAPI dependencies: agent access and optional bearer authentication."""

from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.agents.agent import HybridRAGAgent
from app.config.settings import Settings

# ``auto_error=False`` lets requests without a header reach our check, so
# deployments with no configured token stay open (dev mode) while configured
# ones return a clean 401.
_bearer = HTTPBearer(auto_error=False)


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_agent(request: Request) -> HybridRAGAgent:
    agent: Optional[HybridRAGAgent] = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent is not initialized yet",
        )
    return agent


def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    settings: Settings = Depends(get_settings_dep),
) -> None:
    """Enforce the static bearer token when ``API__AUTH_TOKEN`` is set."""
    expected = settings.api.auth_token
    if expected is None:
        return  # auth disabled by configuration
    supplied = credentials.credentials if credentials else None
    if supplied != expected.get_secret_value():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token",
        )
