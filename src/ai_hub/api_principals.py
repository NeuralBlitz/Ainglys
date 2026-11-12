"""Environment-provisioned, scoped API principals for small Ainglys deployments.

An optional JSON keyring enables credential revocation once the running
process receives updated configuration, without persisting plaintext tokens
or using the legacy shared master key.
This is a bridge to an external identity system, not production IAM.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
import secrets

SCOPES = frozenset({"read", "predict", "train", "models:write"})
_CLIENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_HEX_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
MAX_PRINCIPALS = 32
MAX_CONFIG_BYTES = 32_768


class IdentityConfigError(ValueError):
    """Identity settings cannot be parsed or safely enforced."""


@dataclass(frozen=True)
class APIPrincipal:
    client_id: str
    scopes: frozenset[str]
    fingerprint: str
    legacy: bool = False


def token_sha256(api_key: str) -> str:
    """Fingerprint a high-entropy API key; never persist plaintext."""
    if not isinstance(api_key, str) or not 32 <= len(api_key) <= 1024:
        raise ValueError("API keys must be 32-1024 characters")
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def load_keyring() -> list[APIPrincipal] | None:
    """Read and validate the complete keyring at each request for live revocation.

    If the variable is defined (even empty/invalid), the legacy secret is
    deliberately disabled. An invalid registry must never fall back to it.
    """
    raw = os.environ.get("AI_HUB_API_KEYS_JSON")
    if raw is None:
        return None
    if len(raw.encode("utf-8")) > MAX_CONFIG_BYTES:
        raise IdentityConfigError("Credential registry is too large")
    try:
        rows = json.loads(raw)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise IdentityConfigError("Credential registry is invalid JSON") from exc
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_PRINCIPALS:
        raise IdentityConfigError("Credential registry requires 1-32 clients")
    ids: set[str] = set()
    fingerprints: set[str] = set()
    principals: list[APIPrincipal] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "sha256", "scopes"}:
            raise IdentityConfigError("Credential entry has invalid fields")
        client_id, fingerprint, scopes = row["id"], row["sha256"], row["scopes"]
        if not isinstance(client_id, str) or not _CLIENT_ID.fullmatch(client_id):
            raise IdentityConfigError("Credential client ID is invalid")
        if not isinstance(fingerprint, str) or not _HEX_DIGEST.fullmatch(fingerprint):
            raise IdentityConfigError("Credential SHA256 digest must be lowercase hex")
        if not isinstance(scopes, list) or not scopes or len(scopes) > len(SCOPES):
            raise IdentityConfigError("Credential scope list is invalid")
        if any(not isinstance(scope, str) or scope not in SCOPES for scope in scopes):
            raise IdentityConfigError("Credential scope is not supported")
        if len(set(scopes)) != len(scopes):
            raise IdentityConfigError("Duplicate credential scope")
        if client_id in ids or fingerprint in fingerprints:
            raise IdentityConfigError("Duplicate credential identity")
        ids.add(client_id)
        fingerprints.add(fingerprint)
        principals.append(APIPrincipal(client_id, frozenset(scopes), fingerprint))
    return principals


def authenticate(api_key: str | None) -> APIPrincipal | None:
    """Match a token against a valid registry; return None on wrong credentials.

    Raises IdentityConfigError when configuration cannot be enforced.
    """
    registry = load_keyring()
    if registry is None:
        legacy = os.environ.get("AI_HUB_API_KEY", "")
        try:
            fingerprint = token_sha256(legacy)
        except ValueError as exc:
            raise IdentityConfigError("Legacy API key is not configured") from exc
        if not api_key or not secrets.compare_digest(api_key, legacy):
            return None
        return APIPrincipal("legacy", SCOPES, fingerprint, legacy=True)

    if not api_key or not 32 <= len(api_key) <= 1024:
        return None
    fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    match: APIPrincipal | None = None
    # Evaluate all configured credentials even when one matches.
    for principal in registry:
        if secrets.compare_digest(principal.fingerprint, fingerprint):
            match = principal
    return match


def required_scope(method: str, path: str) -> str | None:
    """Route-to-scope policy. Unknown protected routes must fail closed."""
    if method == "GET" and path in {"/status", "/models"}:
        return "read"
    if method == "POST" and path == "/predict":
        return "predict"
    if method == "POST" and path in {"/train", "/train/supervised"}:
        return "train"
    if method == "POST" and (path == "/save" or path.startswith("/load/")):
        return "models:write"
    if method == "DELETE" and path.startswith("/models/"):
        return "models:write"
    return None
