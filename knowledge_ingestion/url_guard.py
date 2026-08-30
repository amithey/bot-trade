"""Refuse URLs that point back into the deployment's own network.

The article scraper fetches whatever URL a user types into the Knowledge
page. On a hosted deployment that makes the server an HTTP proxy the user
controls: ``169.254.169.254`` returns cloud instance credentials, and an
internal admin panel on the same VPC returns exactly the HTML the scraper is
looking for, which then lands in the knowledge base and becomes readable
through RAG.

Two things have to be checked, and checking only the first is the usual way
this ends up broken:

1. The hostname in the URL, resolved to an address.
2. Every address it resolves to — a name under the attacker's control can
   simply have an A record pointing at 127.0.0.1.

Redirects need the same treatment, which is why the caller must disable
automatic redirect following and re-validate each hop.
"""
from __future__ import annotations

import ipaddress
import socket
from typing import Optional
from urllib.parse import urlparse


class BlockedURLError(ValueError):
    """Raised for a URL that resolves somewhere it must not be fetched from."""


#: Ports worth blocking outright even on a public address — reaching a
#: database or a mail relay through this scraper is never the intent.
_BLOCKED_PORTS = {22, 23, 25, 445, 3306, 5432, 6379, 9200, 11211, 27017}


def _address_is_internal(addr: str) -> bool:
    """True for anything that is not a normal public internet address."""
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return True  # unparseable — refuse rather than guess

    # `is_private` alone is not enough: it misses the cloud metadata endpoint
    # (link-local) and the unspecified address, both of which are the point.
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local        # 169.254.0.0/16 — cloud metadata lives here
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
        or (getattr(ip, "is_site_local", False))
    )


def _resolve(host: str) -> list[str]:
    """Every address *host* resolves to, v4 and v6."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise BlockedURLError(f"could not resolve host {host!r}: {exc}") from exc
    return sorted({info[4][0] for info in infos})


def check_url(url: str, *, resolver=None) -> str:
    """Validate *url* for outbound fetching, returning it unchanged.

    Raises :class:`BlockedURLError` when the scheme is not http(s), the port
    is one that has no business being scraped, or the host resolves to any
    address inside the deployment's own network.

    ``resolver`` is injectable so the rule can be tested without DNS.
    """
    parsed = urlparse((url or "").strip())

    if parsed.scheme not in ("http", "https"):
        raise BlockedURLError(
            f"only http(s) URLs can be ingested, got {parsed.scheme or 'no'} "
            f"scheme")
    if not parsed.hostname:
        raise BlockedURLError(f"no host in URL: {url!r}")

    if parsed.port is not None and parsed.port in _BLOCKED_PORTS:
        raise BlockedURLError(f"port {parsed.port} is not fetchable")

    resolve = resolver or _resolve
    addresses = resolve(parsed.hostname)
    if not addresses:
        raise BlockedURLError(f"host {parsed.hostname!r} resolved to nothing")

    # Every address, not just the first: a host that resolves to both a public
    # and a private address must still be refused.
    for addr in addresses:
        if _address_is_internal(addr):
            raise BlockedURLError(
                f"{parsed.hostname!r} resolves to {addr}, which is inside the "
                f"deployment's own network — refusing to fetch it"
            )
    return url


def is_allowed(url: str, *, resolver=None) -> tuple[bool, Optional[str]]:
    """Non-raising form: ``(ok, reason_if_blocked)``."""
    try:
        check_url(url, resolver=resolver)
    except BlockedURLError as exc:
        return False, str(exc)
    return True, None
