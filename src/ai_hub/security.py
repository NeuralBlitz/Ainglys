"""
Security & Authentication for AI Hub API

Adds OAuth2, JWT tokens, and API key authentication.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, List
from functools import wraps
import hashlib
import secrets
import jwt
from fastapi import HTTPException, Security, Depends
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader, HTTPBearer
from pydantic import BaseModel


# Security configuration
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Security schemes
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
http_bearer = HTTPBearer(auto_error=False)


class User(BaseModel):
    """User model."""

    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False
    role: str = "user"  # user, admin, researcher
    scopes: List[str] = []


class UserInDB(User):
    """User with hashed password."""

    hashed_password: str


class Token(BaseModel):
    """Token response."""

    access_token: str
    token_type: str
    expires_in: int


class TokenData(BaseModel):
    """Token payload."""

    username: Optional[str] = None
    scopes: List[str] = []


# Mock database (replace with real DB in production)
fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Administrator",
        "email": "admin@aihub.com",
        "hashed_password": hashlib.sha256("admin123".encode()).hexdigest(),
        "disabled": False,
        "role": "admin",
        "scopes": ["read", "write", "admin", "delete"],
    },
    "researcher": {
        "username": "researcher",
        "full_name": "Research User",
        "email": "researcher@aihub.com",
        "hashed_password": hashlib.sha256("research123".encode()).hexdigest(),
        "disabled": False,
        "role": "researcher",
        "scopes": ["read", "write", "train", "predict"],
    },
    "user": {
        "username": "user",
        "full_name": "Standard User",
        "email": "user@aihub.com",
        "hashed_password": hashlib.sha256("user123".encode()).hexdigest(),
        "disabled": False,
        "role": "user",
        "scopes": ["read", "predict"],
    },
}

# API Keys storage
api_keys_db: Dict[str, Dict] = {}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password


def get_user(db, username: str) -> Optional[UserInDB]:
    """Get user from database."""
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    return None


def authenticate_user(db, username: str, password: str) -> Optional[UserInDB]:
    """Authenticate user credentials."""
    user = get_user(db, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> TokenData:
    """Verify JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        scopes: List[str] = payload.get("scopes", [])

        if username is None:
            raise HTTPException(
                status_code=401, detail="Invalid authentication credentials"
            )

        return TokenData(username=username, scopes=scopes)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(
            status_code=401, detail="Invalid authentication credentials"
        )


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    api_key: Optional[str] = Depends(api_key_header),
) -> User:
    """
    Get current user from token or API key.

    Supports both OAuth2 tokens and API keys.
    """
    credentials_exception = HTTPException(
        status_code=401,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Try OAuth2 token first
    if token:
        token_data = verify_token(token)
        user = get_user(fake_users_db, token_data.username)
        if user is None:
            raise credentials_exception
        return user

    # Try API key
    if api_key:
        if api_key not in api_keys_db:
            raise credentials_exception

        key_data = api_keys_db[api_key]
        if not key_data.get("active", False):
            raise HTTPException(status_code=403, detail="API key is disabled")

        user = get_user(fake_users_db, key_data["username"])
        if user is None:
            raise credentials_exception
        return user

    raise credentials_exception


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get current active user."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def check_scope(required_scope: str):
    """Decorator to check user scope."""

    async def scope_checker(current_user: User = Depends(get_current_active_user)):
        if (
            required_scope not in current_user.scopes
            and "admin" not in current_user.scopes
        ):
            raise HTTPException(
                status_code=403,
                detail=f"Not enough permissions. Required scope: {required_scope}",
            )
        return current_user

    return scope_checker


def require_role(required_role: str):
    """Decorator to require specific role."""

    async def role_checker(current_user: User = Depends(get_current_active_user)):
        role_hierarchy = {"user": 1, "researcher": 2, "admin": 3}

        user_level = role_hierarchy.get(current_user.role, 0)
        required_level = role_hierarchy.get(required_role, 0)

        if user_level < required_level:
            raise HTTPException(
                status_code=403,
                detail=f"Not enough permissions. Required role: {required_role}",
            )
        return current_user

    return role_checker


# API Key management


def generate_api_key(username: str, name: str = "default") -> str:
    """Generate new API key for user."""
    api_key = f"ak_{secrets.token_urlsafe(32)}"

    api_keys_db[api_key] = {
        "username": username,
        "name": name,
        "created_at": datetime.utcnow().isoformat(),
        "active": True,
        "last_used": None,
        "usage_count": 0,
    }

    return api_key


def revoke_api_key(api_key: str) -> bool:
    """Revoke an API key."""
    if api_key in api_keys_db:
        api_keys_db[api_key]["active"] = False
        return True
    return False


def get_user_api_keys(username: str) -> List[Dict]:
    """Get all API keys for a user."""
    return [
        {"key": k, **v} for k, v in api_keys_db.items() if v["username"] == username
    ]


def log_api_key_usage(api_key: str):
    """Log API key usage."""
    if api_key in api_keys_db:
        api_keys_db[api_key]["last_used"] = datetime.utcnow().isoformat()
        api_keys_db[api_key]["usage_count"] += 1


# Rate limiting (simple in-memory implementation)
rate_limit_db: Dict[str, Dict] = {}


def check_rate_limit(
    identifier: str, max_requests: int = 100, window_seconds: int = 3600
) -> bool:
    """
    Check if request is within rate limit.

    Args:
        identifier: User or API key identifier
        max_requests: Maximum requests allowed in window
        window_seconds: Time window in seconds

    Returns:
        True if allowed, False if rate limited
    """
    now = datetime.utcnow()

    if identifier not in rate_limit_db:
        rate_limit_db[identifier] = {"requests": 1, "window_start": now}
        return True

    data = rate_limit_db[identifier]
    window_start = data["window_start"]

    # Reset window if expired
    if (now - window_start).total_seconds() > window_seconds:
        data["requests"] = 1
        data["window_start"] = now
        return True

    # Check limit
    if data["requests"] >= max_requests:
        return False

    data["requests"] += 1
    return True


# Security headers middleware
async def security_headers_middleware(request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    response.headers["Content-Security-Policy"] = "default-src 'self'"

    return response


# Input sanitization
def sanitize_path(path: str) -> str:
    """Sanitize file path to prevent directory traversal."""
    # Remove null bytes
    path = path.replace("\x00", "")

    # Normalize path
    import os.path

    path = os.path.normpath(path)

    # Check for directory traversal
    if ".." in path or path.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid path")

    return path


def secure_torch_load(filepath: str, map_location="cpu"):
    """
    Securely load PyTorch model.

    Uses weights_only=True to prevent code execution attacks.
    """
    try:
        # PyTorch 2.0+ supports weights_only
        return torch.load(filepath, map_location=map_location, weights_only=True)
    except TypeError:
        # Fallback for older versions
        return torch.load(filepath, map_location=map_location)


# Permission decorators for common operations
require_read = check_scope("read")
require_write = check_scope("write")
require_train = check_scope("train")
require_predict = check_scope("predict")
require_delete = check_scope("delete")
require_admin = require_role("admin")


# Export commonly used items
__all__ = [
    "oauth2_scheme",
    "api_key_header",
    "get_current_user",
    "get_current_active_user",
    "authenticate_user",
    "create_access_token",
    "verify_token",
    "Token",
    "User",
    "generate_api_key",
    "revoke_api_key",
    "check_rate_limit",
    "sanitize_path",
    "secure_torch_load",
    "require_read",
    "require_write",
    "require_train",
    "require_predict",
    "require_delete",
    "require_admin",
]
