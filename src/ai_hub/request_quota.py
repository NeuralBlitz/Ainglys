"""Single-process rate budgets for authenticated Ainglys API requests.

This provides a bounded, thread-safe guard for the current single API process.
It is *not* a distributed quota system or a substitute for identity/RBAC.
"""
from __future__ import annotations

from collections import deque
import hashlib
import math
import threading
import time
from typing import Callable

WINDOW_SECONDS = 60


class RequestQuota:
    """A moving-window limiter shared by the currently configured API key."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._fingerprint: bytes | None = None
        self._all_requests: deque[float] = deque()
        self._training_requests: deque[float] = deque()

    def reset(self) -> None:
        """Clear records (useful for deterministic tests)."""
        with self._lock:
            self._fingerprint = None
            self._all_requests.clear()
            self._training_requests.clear()

    @staticmethod
    def _prune(records: deque[float], now: float) -> None:
        while records and now - records[0] >= WINDOW_SECONDS:
            records.popleft()

    def check(
        self,
        api_key: str,
        *,
        training: bool,
        request_limit: int,
        training_limit: int,
    ) -> int | None:
        """Consume one request budget slot, or return Retry-After seconds.

        The raw API key is never persisted. A rotated key begins a fresh window.
        Calls are atomic under one process-local mutex.
        """
        if not (1 <= request_limit <= 10000 and 1 <= training_limit <= request_limit):
            raise ValueError("Quota limits are invalid")
        fingerprint = hashlib.sha256(api_key.encode("utf-8")).digest()
        now = self._clock()
        with self._lock:
            if self._fingerprint != fingerprint:
                self._fingerprint = fingerprint
                self._all_requests.clear()
                self._training_requests.clear()
            self._prune(self._all_requests, now)
            self._prune(self._training_requests, now)
            blocked = []
            if len(self._all_requests) >= request_limit:
                blocked.append(self._all_requests[0] + WINDOW_SECONDS - now)
            if training and len(self._training_requests) >= training_limit:
                blocked.append(self._training_requests[0] + WINDOW_SECONDS - now)
            if blocked:
                return max(1, math.ceil(max(blocked)))
            self._all_requests.append(now)
            if training:
                self._training_requests.append(now)
            return None


request_quota = RequestQuota()


class PrincipalRequestQuotas:
    """Separate process-local windows for up to 32 configured client tokens.

    Stale entries expire; never evict an active quota merely to admit another
    credential, which could let rotating identities bypass rate limits.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: dict[bytes, tuple[RequestQuota, float]] = {}

    def reset(self):
        with self._lock:
            self._entries.clear()

    def check(
        self,
        token: str,
        *,
        training: bool,
        request_limit: int,
        training_limit: int,
    ) -> int | None:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        now = self._clock()
        with self._lock:
            # Keep last-used time on active clients, but do not forgive any
            # requests within the fixed rolling 60-second window.
            stale = [
                fingerprint for fingerprint, (_, last_seen) in self._entries.items()
                if now - last_seen >= WINDOW_SECONDS
            ]
            for fingerprint in stale:
                del self._entries[fingerprint]
            if digest not in self._entries:
                if len(self._entries) >= 32:
                    return WINDOW_SECONDS
                self._entries[digest] = (RequestQuota(self._clock), now)
            quota, _ = self._entries[digest]
            self._entries[digest] = (quota, now)
        return quota.check(
            token, training=training,
            request_limit=request_limit, training_limit=training_limit,
        )


principal_request_quotas = PrincipalRequestQuotas()
