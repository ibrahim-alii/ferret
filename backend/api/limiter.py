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

    Behind a reverse proxy/load balancer, ``request.client.host`` is the proxy
    address, so every user would share one bucket. ``X-Forwarded-For`` carries the
    chain, but the *left-most* hops are client-supplied and trivially spoofed — an
    attacker can rotate them to dodge the limit. Only the right-most entries, appended
    by infrastructure we control, are trustworthy.

    Set ``TRUSTED_PROXY_COUNT`` to the number of proxies in front of the app. With N
    trusted proxies the real client IP is the Nth entry from the right of XFF. When it
    is 0 (the default) we never trust the header and use the socket peer address, so a
    misconfigured deploy fails closed rather than honoring a spoofable header.
    """
    trusted = int(os.environ.get("TRUSTED_PROXY_COUNT", "0"))
    if trusted > 0:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            hops = [h.strip() for h in forwarded.split(",") if h.strip()]
            if len(hops) >= trusted:
                return hops[-trusted]
    return get_remote_address(request)


# `enabled` is off under tests (DISABLE_RATE_LIMIT set in tests/conftest.py) so the
# shared in-memory window doesn't bleed counts across cases.
limiter = Limiter(
    key_func=_client_ip,
    enabled=not os.environ.get("DISABLE_RATE_LIMIT"),
)
