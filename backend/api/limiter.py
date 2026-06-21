"""Shared slowapi Limiter instance.

Lives in its own module so routers can import it without a circular import
back into backend.api.app (which imports the routers).
"""

import os

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request


def _client_ip(request: Request) -> str:
    """Real client IP for per-user rate limiting.

    Behind Fly.io's proxy, ``request.client.host`` is the proxy address, so every
    user would share a single bucket. Fly sets ``Fly-Client-IP`` to the true
    client IP (and overwrites any client-supplied value, so it can't be spoofed);
    fall back to the left-most ``X-Forwarded-For`` hop, then the socket address.
    """
    fly_ip = request.headers.get("fly-client-ip")
    if fly_ip:
        return fly_ip
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


# `enabled` is off under tests (DISABLE_RATE_LIMIT set in tests/conftest.py) so the
# shared in-memory window doesn't bleed counts across cases.
limiter = Limiter(
    key_func=_client_ip,
    enabled=not os.environ.get("DISABLE_RATE_LIMIT"),
)
