import asyncio
import json
import logging
import random
import time
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from urllib.robotparser import RobotFileParser
from pathlib import Path
from typing import AsyncGenerator, Dict, Set, Optional, List

import httpx
from selectolax.parser import HTMLParser

USER_AGENT = "FindexCrawler/1.0 (+https://github.com/rmalkevy/Programming-Practice-Projects)"

logging.basicConfig(
    filename="crawl.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8"
)

def canonicalize_url(url: str) -> str:
    """Канонізація URL: схема/хост у нижній регістр, відкидання фрагмента, сортування query."""
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    # Сортування query параметрів
    query_params = parse_qsl(parsed.query, keep_blank_values=True)
    query_params.sort()
    query = urlencode(query_params)

    return urlunparse((scheme, netloc, path, "", query, ""))


class RobotsCache:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.cache: Dict[str, RobotFileParser] = {}

    async def is_allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        host = f"{parsed.scheme}://{parsed.netloc}"
        if host not in self.cache:
            rfp = RobotFileParser()
            robots_url = f"{host}/robots.txt"
            try:
                resp = await self.client.get(robots_url, timeout=5.0)
                if resp.status_code == 200:
                    rfp.parse(resp.text.splitlines())
                else:
                    rfp.allow_all = True
            except Exception:
                rfp.allow_all = True
            self.cache[host] = rfp
        return self.cache[host].can_fetch(USER_AGENT, url)


async def fetch_with_retry(
    client: httpx.AsyncClient,
    url: str,
    max_retries: int = 3
) -> Optional[httpx.Response]:
    """Fetch з підтримкою exponential backoff + jitter та обробкою Retry-After (сумісно з Python 3.8)."""
    for attempt in range(max_retries + 1):
        try:
            # Передаємо timeout безпосередньо у виклик get для сумісності з Python < 3.11
            response = await client.get(
                url,
                headers={"User-Agent": USER_AGENT},
                follow_redirects=True,
                timeout=10.0
            )
            if response.status_code == 200:
                logging.info(f"SUCCESS {response.status_code}: {url}")
                return response
            elif response.status_code in (429,) or response.status_code >= 500:
                logging.warning(f"RETRY {response.status_code}: {url} (Attempt {attempt+1})")
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    backoff = float(retry_after)
                else:
                    backoff = (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(backoff)
            else:
                logging.error(f"FAIL {response.status_code}: {url}")
                return None
        except (httpx.TimeoutException, httpx.RequestError) as e:
            logging.warning(f"TIMEOUT/ERROR: {url} - {str(e)} (Attempt {attempt+1})")
            if attempt < max_retries:
                backoff = (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(backoff)
            else:
                logging.error(f"MAX RETRIES EXCEEDED: {url}")
                return None
    return None


async def crawl(
    start_url: str,
    max_pages: int = 200,
    concurrency: int = 5,
    delay_per_host: float = 0.1
) -> AsyncGenerator[dict, None]:
    """Асинхронний генератор краулера сторінок."""
    start_url = canonicalize_url(start_url)
    seen: Set[str] = {start_url}
    queue: asyncio.Queue[str] = asyncio.Queue()
    await queue.put(start_url)

    global_sem = asyncio.Semaphore(concurrency)
    host_sems: Dict[str, asyncio.Semaphore] = {}
    host_last_request: Dict[str, float] = {}

    pages_fetched = 0

    async with httpx.AsyncClient() as client:
        robots = RobotsCache(client)

        while not queue.empty() and pages_fetched < max_pages:
            url = await queue.get()

            if not await robots.is_allowed(url):
                logging.info(f"BLOCKED BY ROBOTS.TXT: {url}")
                queue.task_done()
                continue

            parsed = urlparse(url)
            host = parsed.netloc

            if host not in host_sems:
                host_sems[host] = asyncio.Semaphore(1)

            async with global_sem, host_sems[host]:
                # Пауза між запитами до того самого хоста
                now = time.time()
                last = host_last_request.get(host, 0)
                if now - last < delay_per_host:
                    await asyncio.sleep(delay_per_host - (now - last))
                host_last_request[host] = time.time()

                resp = await fetch_with_retry(client, url)
                queue.task_done()

                if not resp or "text/html" not in resp.headers.get("content-type", ""):
                    continue

                html = resp.text
                tree = HTMLParser(html)

                # Витяг текстів
                text = tree.body.text(strip=True) if tree.body else ""

                pages_fetched += 1
                page_data = {
                    "url": url,
                    "title": tree.css_first("title").text() if tree.css_first("title") else "",
                    "text": text,
                    "status": resp.status_code
                }

                # Витяг нових посилань
                for a in tree.css("a[href]"):
                    href = a.attributes.get("href")
                    if not href:
                        continue
                    full_url = httpx.URL(url).join(href)
                    canon_url = canonicalize_url(str(full_url))

                    # Обмеження обходу лише поточним хостом
                    if urlparse(canon_url).netloc == host and canon_url not in seen:
                        seen.add(canon_url)
                        await queue.put(canon_url)

                yield page_data