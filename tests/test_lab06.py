import pytest
import httpx
import asyncio
from findex.crawler import canonicalize_url, RobotsCache, fetch_with_retry


def test_canonicalize_url():
    url = "HTTP://Example.com:80/docs/../index.html?b=2&a=1#section"
    canon = canonicalize_url(url)
    assert "http://example.com" in canon
    assert "#section" not in canon
    assert "a=1&b=2" in canon


def test_robots_cache():
    async def run():
        def handler(request):
            if request.url.path == "/robots.txt":
                return httpx.Response(200, text="User-agent: *\nDisallow: /private/")
            return httpx.Response(200, text="OK")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            robots = RobotsCache(client)
            assert await robots.is_allowed("http://example.com/public") is True
            assert await robots.is_allowed("http://example.com/private/data") is False

    asyncio.run(run())


def test_fetch_with_retry_success():
    async def run():
        def handler(request):
            return httpx.Response(200, text="Success")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await fetch_with_retry(client, "http://example.com")
            assert resp is not None
            assert resp.status_code == 200

    asyncio.run(run())