"""
Tests for knowledge_ingestion/url_guard.py and the SSRF defence in the
article scraper.

The scraper fetches a URL the user types into the Knowledge page. On a hosted
deployment that turns the server into an HTTP proxy the user aims: the cloud
metadata endpoint hands out instance credentials, and an internal admin panel
returns exactly the HTML the scraper wants — which then lands in the shared
knowledge base and becomes readable through RAG.

DNS is injected throughout, so nothing here resolves a real hostname.
"""
from __future__ import annotations

import pytest

from knowledge_ingestion.url_guard import (
    BlockedURLError,
    check_url,
    is_allowed,
)


def _resolves_to(*addresses):
    """A resolver stand-in returning fixed addresses."""
    return lambda host: list(addresses)


# --------------------------------------------------------------------------- #
# Scheme and shape
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("url", [
    "ftp://example.com/x",
    "file:///etc/passwd",
    "gopher://example.com",
    "javascript:alert(1)",
    "",
])
def test_non_http_schemes_are_refused(url):
    with pytest.raises(BlockedURLError):
        check_url(url, resolver=_resolves_to("93.184.216.34"))


def test_url_without_a_host_is_refused():
    with pytest.raises(BlockedURLError, match="no host"):
        check_url("http:///just-a-path", resolver=_resolves_to("93.184.216.34"))


# --------------------------------------------------------------------------- #
# The addresses that matter
# --------------------------------------------------------------------------- #
def test_cloud_metadata_endpoint_is_refused():
    """169.254.169.254 returns instance credentials on AWS/GCP/Azure."""
    with pytest.raises(BlockedURLError, match="own network"):
        check_url("http://169.254.169.254/latest/meta-data/",
                  resolver=_resolves_to("169.254.169.254"))


@pytest.mark.parametrize("addr", [
    "127.0.0.1",        # loopback
    "10.0.0.5",         # private
    "172.16.4.2",       # private
    "192.168.1.1",      # private
    "169.254.169.254",  # link-local / metadata
    "0.0.0.0",          # unspecified
    "::1",              # IPv6 loopback
    "fd00::1",          # IPv6 unique-local
    "fe80::1",          # IPv6 link-local
])
def test_internal_addresses_are_refused(addr):
    with pytest.raises(BlockedURLError):
        check_url("http://internal.example/", resolver=_resolves_to(addr))


def test_a_normal_public_address_is_allowed():
    assert check_url("https://www.investopedia.com/terms/m/macd.asp",
                     resolver=_resolves_to("93.184.216.34"))


# --------------------------------------------------------------------------- #
# DNS rebinding — the reason the check is on the resolved address
# --------------------------------------------------------------------------- #
def test_a_public_looking_hostname_that_resolves_inward_is_refused():
    """Checking the hostname string alone is the usual way this stays broken:
    an attacker just points their own domain at 127.0.0.1."""
    with pytest.raises(BlockedURLError, match="own network"):
        check_url("http://totally-normal-blog.com/post",
                  resolver=_resolves_to("127.0.0.1"))


def test_every_resolved_address_is_checked_not_just_the_first():
    """A host answering with one public and one private address must still be
    refused — otherwise which one gets connected to is a race."""
    with pytest.raises(BlockedURLError):
        check_url("http://mixed.example/",
                  resolver=_resolves_to("93.184.216.34", "10.0.0.5"))


def test_unresolvable_host_is_refused():
    def _fail(host):
        raise BlockedURLError("nope")
    with pytest.raises(BlockedURLError):
        check_url("http://nx.example/", resolver=_fail)


def test_empty_resolution_is_refused():
    with pytest.raises(BlockedURLError, match="resolved to nothing"):
        check_url("http://void.example/", resolver=lambda h: [])


# --------------------------------------------------------------------------- #
# Ports
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("port", [22, 3306, 5432, 6379, 27017])
def test_service_ports_are_refused_even_on_a_public_address(port):
    with pytest.raises(BlockedURLError, match="not fetchable"):
        check_url(f"http://example.com:{port}/",
                  resolver=_resolves_to("93.184.216.34"))


def test_ordinary_web_ports_are_fine():
    assert check_url("http://example.com:8080/",
                     resolver=_resolves_to("93.184.216.34"))


# --------------------------------------------------------------------------- #
# is_allowed
# --------------------------------------------------------------------------- #
def test_is_allowed_reports_a_reason_instead_of_raising():
    ok, reason = is_allowed("http://10.0.0.1/", resolver=_resolves_to("10.0.0.1"))
    assert ok is False
    assert "own network" in reason


def test_is_allowed_on_a_good_url():
    ok, reason = is_allowed("https://example.com/",
                            resolver=_resolves_to("93.184.216.34"))
    assert ok is True and reason is None


# --------------------------------------------------------------------------- #
# The scraper applies it, including on redirects
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, status=200, location=None, text="<html><p>ok</p></html>"):
        self.status_code = status
        self.headers = {"Content-Type": "text/html"}
        if location:
            self.headers["Location"] = location
        self.text = text

    @property
    def is_redirect(self):
        return self.status_code in (301, 302, 303, 307, 308)

    is_permanent_redirect = is_redirect

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def test_scraper_blocks_an_internal_url_without_requesting_it(monkeypatch):
    import knowledge_ingestion.article_scraper as art
    from knowledge_ingestion.article_scraper import ArticleScraper

    called = []
    monkeypatch.setattr(art.requests, "get",
                        lambda *a, **kw: called.append(1) or _Resp())
    monkeypatch.setattr(
        "knowledge_ingestion.url_guard._resolve", lambda host: ["169.254.169.254"]
    )
    with pytest.raises(BlockedURLError):
        ArticleScraper._fetch("http://169.254.169.254/latest/meta-data/")
    assert called == [], "the request must not be made at all"


def test_scraper_revalidates_every_redirect_hop(monkeypatch):
    """The classic bypass: a permitted public URL that 302s inward.

    `requests` follows redirects automatically, so a check done only on the
    URL the user typed would never see the metadata endpoint.
    """
    import knowledge_ingestion.article_scraper as art
    from knowledge_ingestion.article_scraper import ArticleScraper

    def _fake_get(url, **kw):
        if "public" in url:
            return _Resp(302, location="http://169.254.169.254/latest/meta-data/")
        return _Resp(200)

    monkeypatch.setattr(art.requests, "get", _fake_get)
    monkeypatch.setattr(
        "knowledge_ingestion.url_guard._resolve",
        lambda host: ["169.254.169.254"] if "169.254" in host else ["93.184.216.34"],
    )
    with pytest.raises(BlockedURLError, match="own network"):
        ArticleScraper._fetch("http://public.example/article")


def test_scraper_follows_a_legitimate_redirect(monkeypatch):
    import knowledge_ingestion.article_scraper as art
    from knowledge_ingestion.article_scraper import ArticleScraper

    def _fake_get(url, **kw):
        if "old" in url:
            return _Resp(301, location="https://new.example/article")
        return _Resp(200, text="<html><p>the real article</p></html>")

    monkeypatch.setattr(art.requests, "get", _fake_get)
    monkeypatch.setattr("knowledge_ingestion.url_guard._resolve",
                        lambda host: ["93.184.216.34"])
    html = ArticleScraper._fetch("https://old.example/article")
    assert "the real article" in html


def test_scraper_gives_up_on_a_redirect_loop(monkeypatch):
    import knowledge_ingestion.article_scraper as art
    from knowledge_ingestion.article_scraper import ArticleScraper

    monkeypatch.setattr(
        art.requests, "get",
        lambda url, **kw: _Resp(302, location="https://loop.example/again"),
    )
    monkeypatch.setattr("knowledge_ingestion.url_guard._resolve",
                        lambda host: ["93.184.216.34"])
    with pytest.raises(ValueError, match="Too many redirects"):
        ArticleScraper._fetch("https://loop.example/start")
