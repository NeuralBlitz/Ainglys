"""Fail-closed API-key authentication and model identifier validation."""
import os
import re

from fastapi import HTTPException, Request, Security

from .request_quota import request_quota, principal_request_quotas
from .api_principals import authenticate, required_scope, IdentityConfigError
from fastapi.security import APIKeyHeader

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


def require_api_key(
    request: Request, provided: str | None = Security(_api_key_header)
) -> None:
    """Authenticate, authorize route scope, then consume a per-key quota slot."""
    try:
        principal = authenticate(provided)
    except IdentityConfigError:
        raise HTTPException(
            status_code=503, detail="API identity configuration is invalid"
        )

    if principal is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "APIKey"},
        )

    required = required_scope(request.method, request.url.path)
    if required is None or required not in principal.scopes:
        raise HTTPException(status_code=403, detail="Insufficient API permissions")

    try:
        request_limit = int(os.environ.get("AI_HUB_REQUESTS_PER_MINUTE", "600"))
        training_limit = int(os.environ.get("AI_HUB_TRAININGS_PER_MINUTE", "60"))
        if not (1 <= training_limit <= request_limit <= 10000):
            raise ValueError("quota settings out of range")
    except ValueError:
        raise HTTPException(status_code=503, detail="API quota configuration is invalid")

    training = required == "train"
    if principal.legacy:
        # Preserve the existing single-key quota behavior for deployments
        # without AI_HUB_API_KEYS_JSON.
        wait = request_quota.check(
            provided, training=training,
            request_limit=request_limit, training_limit=training_limit,
        )
    else:
        wait = principal_request_quotas.check(
            provided, training=training,
            request_limit=request_limit, training_limit=training_limit,
        )
    if wait is not None:
        raise HTTPException(
            status_code=429,
            detail="API request quota exceeded",
            headers={"Retry-After": str(wait)},
        )
    # Record only the public client identifier, never the raw API token.
    request.state.api_client_id = principal.client_id


def validate_model_name(name: str) -> str:
    """Reject path separators, traversal components and ambiguous model names."""
    if not _MODEL_NAME.fullmatch(name):
        raise HTTPException(
            status_code=422,
            detail="Model name must be 1-64 ASCII letters, numbers, '-' or '_', "
                   "starting with a letter or number",
        )
    return name
