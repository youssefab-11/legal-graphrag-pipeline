"""Specialized scraper for qanoon.om and decree.om.

Extracts Omani legal documents in Arabic and English, converts them to
Markdown, and prepares them for ingestion into the GraphRAG pipeline.

Designed for 100% coverage of qanoon.om. The scraper uses lightweight
``requests`` for listing and document pages and falls back to Playwright
only when a page cannot be retrieved with plain HTTP. Document pages are
scraped concurrently with a configurable worker pool to maximize throughput.
"""

import json
import logging
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, Page, Browser, Response

from src.config.settings import settings
from src.scraper.parser import LegalDocumentParser
from src.scraper.state_manager import StateManager

logger = logging.getLogger(__name__)


class QanoonScraper:
    """Scraper for qanoon.om Arabic and decree.om English legal documents."""

    # Document detail URL pattern: /p/YYYY/slug/
    DOC_URL_PATTERN = re.compile(r"/p/\d{4}/[^/]+/?$")

    # Arabic numerals to Western numerals mapping
    ARABIC_NUMERALS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

    @staticmethod
    def normalize_document_url(url: str) -> str:
        """Normalize a document URL by stripping fragments and ensuring trailing slash.

        qanoon.om sometimes links to the same document with different
        ``#more-XXX`` fragments. This treats them as one document.
        """
        # Remove fragment
        url = url.split("#")[0]
        # Ensure trailing slash
        return url.rstrip("/") + "/"

    def __init__(
        self,
        base_url: str = "https://qanoon.om",
        english_base_url: str = "https://decree.om",
        output_dir: Optional[Path] = None,
        state_manager: Optional[StateManager] = None,
        max_documents: Optional[int] = None,
        delay_min: float = 0.2,
        delay_max: float = 0.7,
        use_playwright_fallback: bool = True,
        max_workers: int = 5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.english_base_url = english_base_url.rstrip("/")
        self.output_dir = output_dir or settings.RAW_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state = state_manager or StateManager(filename="qanoon_state.json")
        self.max_documents = max_documents
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.use_playwright_fallback = use_playwright_fallback
        self.max_workers = max_workers
        self.parser = LegalDocumentParser()

        self.session = requests.Session()
        self.session.headers.update(self._default_headers())

        self.playwright = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self._playwright_started = False
        self._playwright_lock = threading.Lock()

        # Thread-local requests sessions for concurrent workers
        self._local = threading.local()

        # Timing / progress
        self.stats: Dict[str, Any] = {
            "started_at": None,
            "ended_at": None,
            "pages_discovered": 0,
            "docs_scraped": 0,
            "docs_failed": 0,
            "requests_total": 0,
            "requests_failed": 0,
        }
        self._stat_lock = threading.Lock()

    @staticmethod
    def _default_headers() -> Dict[str, str]:
        return {
            "User-Agent": settings.USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "ar,en-US;q=0.9,en;q=0.8",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

    def _get_worker_session(self) -> requests.Session:
        """Get or create a thread-local requests session for a worker thread.

        Creating an SSL context per request is expensive; reusing a session
        per worker thread avoids that overhead and enables connection keep-alive.
        """
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(self._default_headers())
            self._local.session = session
        return session

    def _random_delay(self) -> None:
        """Sleep for a randomized interval."""
        delay = random.uniform(self.delay_min, self.delay_max)
        time.sleep(delay)

    def start_playwright(self, headless: bool = True) -> "QanoonScraper":
        """Initialize Playwright browser on demand."""
        with self._playwright_lock:
            if self._playwright_started:
                return self
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=headless)
            self.page = self.browser.new_page(
                viewport={"width": 1920, "height": 1080},
                user_agent=settings.USER_AGENT,
            )
            self.page.set_extra_http_headers(
                {
                    "Accept-Language": "ar,en;q=0.9",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                }
            )
            self._playwright_started = True
            logger.info("Playwright browser started for fallback scraping.")
        return self

    def stop_playwright(self) -> None:
        """Clean up browser resources."""
        with self._playwright_lock:
            if self.browser:
                self.browser.close()
                self.browser = None
            if self.playwright:
                self.playwright.stop()
                self.playwright = None
            self._playwright_started = False
            logger.info("Playwright browser stopped.")

    def _fetch_with_requests(self, session: requests.Session, url: str, timeout: int = 30) -> Optional[str]:
        """Fetch page HTML using requests."""
        self._random_delay()
        with self._stat_lock:
            self.stats["requests_total"] += 1
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        return response.text

    def _fetch_with_playwright(self, url: str, timeout: int = 60) -> Optional[str]:
        """Fetch page HTML using Playwright."""
        if not self._playwright_started:
            self.start_playwright()
        assert self.page is not None

        self._random_delay()
        response: Response = self.page.goto(url, wait_until="networkidle", timeout=timeout * 1000)
        if response is None or response.status >= 400:
            status = response.status if response else "?"
            raise RuntimeError(f"Playwright HTTP {status}")
        return self.page.content()

    def fetch_page(
        self,
        url: str,
        session: Optional[requests.Session] = None,
        retries: int = 3,
    ) -> Optional[str]:
        """Fetch a page, preferring requests and falling back to Playwright."""
        session = session or self.session
        last_error: Optional[Exception] = None

        for attempt in range(1, retries + 1):
            try:
                return self._fetch_with_requests(session, url)
            except Exception as exc:
                last_error = exc
                logger.debug("requests attempt %d failed for %s: %s", attempt, url, exc)
                time.sleep(random.uniform(0.5, 1.5) * attempt)

        with self._stat_lock:
            self.stats["requests_failed"] += 1

        if self.use_playwright_fallback:
            logger.info("Falling back to Playwright for %s", url)
            for attempt in range(1, retries + 1):
                try:
                    return self._fetch_with_playwright(url)
                except Exception as exc:
                    last_error = exc
                    logger.warning("Playwright attempt %d failed for %s: %s", attempt, url, exc)
                    time.sleep(random.uniform(1, 3) * attempt)

        logger.error("All fetch attempts failed for %s: %s", url, last_error)
        return None

    def discover_document_links(
        self,
        listing_url: str,
        session: Optional[requests.Session] = None,
    ) -> List[str]:
        """Discover document detail URLs from a listing page."""
        html = self.fetch_page(listing_url, session=session)
        if not html:
            return []

        soup = self.parser.create_soup(html)
        links: List[str] = []
        base_netloc = urlparse(self.base_url).netloc

        for a in soup.find_all("a", href=True):
            href = a["href"]
            absolute = urljoin(self.base_url, href)
            parsed = urlparse(absolute)

            if parsed.netloc == base_netloc and self.DOC_URL_PATTERN.match(parsed.path):
                normalized = self.normalize_document_url(absolute)
                if normalized not in links:
                    links.append(normalized)

        logger.info("Discovered %d document links from %s", len(links), listing_url)
        return links

    def get_total_pages(
        self,
        category_url: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ) -> int:
        """Detect the last listing page number from pagination links.

        Returns:
            Total number of listing pages (1-based). Returns 1 on failure.
        """
        base = (category_url or self.base_url).rstrip("/") + "/"
        html = self.fetch_page(base, session=session)
        if not html:
            logger.warning("Could not fetch homepage to detect pagination.")
            return 1

        soup = self.parser.create_soup(html)
        page_numbers: List[int] = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            m = re.search(r"/page/(\d+)/", href)
            if m:
                page_numbers.append(int(m.group(1)))

        if not page_numbers:
            logger.warning("No pagination links found.")
            return 1

        total = max(page_numbers)
        logger.info("Detected %d total listing pages.", total)
        return total

    @staticmethod
    def normalize_arabic_number(text: str) -> str:
        """Convert Arabic-Indic numerals to Western numerals."""
        return text.translate(QanoonScraper.ARABIC_NUMERALS)

    def extract_arabic_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract metadata from Arabic document page."""
        metadata: Dict[str, Any] = {
            "source_url": url,
            "language": "ar",
        }

        title_tag = soup.find("h1", class_="entry-title") or soup.find("h2", class_="entry-title")
        if title_tag:
            metadata["title"] = title_tag.get_text(strip=True)

        cat_tag = soup.find("div", class_="entry-categories")
        if cat_tag:
            cat_link = cat_tag.find("a")
            if cat_link:
                metadata["document_type"] = cat_link.get_text(strip=True)

        date_tag = soup.find("li", class_="post-date")
        if date_tag:
            date_link = date_tag.find("a")
            if date_link:
                metadata["issue_date_text"] = date_link.get_text(strip=True)

        title = metadata.get("title", "")
        number_match = re.search(r"رقم\s*([٠١٢٣٤٥٦٧٨٩\d]+\s*/\s*[٠١٢٣٤٥\u0660-\u0669\d]+)", title)
        if number_match:
            metadata["number"] = self.normalize_arabic_number(number_match.group(1).replace(" ", ""))

        if ":" in title:
            metadata["issuer"] = title.split(":")[0].strip()
        elif "وزارة" in title or "شرطة" in title or "بنك" in title:
            issuer_match = re.match(r"^(وزارة[^:|]+|شرطة[^:|]+|البنك[^:|]+)", title)
            if issuer_match:
                metadata["issuer"] = issuer_match.group(1).strip()

        return metadata

    def extract_english_url(self, soup: BeautifulSoup) -> Optional[str]:
        """Find English version URL from Arabic page."""
        en_link = soup.find("a", class_="decree-link")
        if en_link and en_link.get("href"):
            href = en_link["href"]
            if href.startswith("http"):
                return href
            return urljoin(self.english_base_url, href)
        return None

    def extract_pdf_url(self, soup: BeautifulSoup) -> Optional[str]:
        """Find PDF download URL from document page."""
        pdf_link = soup.find("a", class_="pdf-link")
        if pdf_link and pdf_link.get("href"):
            href = pdf_link["href"]
            if href.startswith("http"):
                return href
            return urljoin(self.base_url, href)
        return None

    def _scrape_arabic_document(
        self,
        session: requests.Session,
        url: str,
    ) -> Optional[Dict[str, Any]]:
        """Scrape Arabic document page with a worker session."""
        html = self.fetch_page(url, session=session)
        if not html:
            return None

        soup = self.parser.create_soup(html)
        metadata = self.extract_arabic_metadata(soup, url)

        content_div = soup.find("div", class_="entry-content")
        if not content_div:
            logger.warning("No entry-content found for %s", url)
            return None

        markdown = self.parser.html_to_markdown(str(content_div), base_url=self.base_url)
        metadata["contentAr"] = markdown
        metadata["raw_markdown"] = markdown
        metadata["english_url"] = self.extract_english_url(soup)
        metadata["pdf_url"] = self.extract_pdf_url(soup)

        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")
        metadata["id"] = "-".join(path_parts[-2:]) if len(path_parts) >= 2 else parsed.path.strip("/").replace("/", "-")

        return metadata

    def _scrape_english_document(
        self,
        session: requests.Session,
        en_url: str,
    ) -> Optional[str]:
        """Scrape English document page and return markdown content."""
        html = self.fetch_page(en_url, session=session)
        if not html:
            return None

        soup = self.parser.create_soup(html)
        content_div = soup.find("div", class_="entry-content")
        if not content_div:
            logger.warning("No entry-content found for English page %s", en_url)
            return None

        return self.parser.html_to_markdown(str(content_div), base_url=self.english_base_url)

    def _save_document(self, doc: Dict[str, Any]) -> Path:
        """Persist a scraped document to disk."""
        file_name = f"{doc['id']}.json"
        file_path = self.output_dir / file_name
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        return file_path

    def _scrape_document_worker(self, url: str) -> Optional[Dict[str, Any]]:
        """Thread worker that scrapes one document."""
        url = self.normalize_document_url(url)

        if self.state.is_completed(url):
            logger.debug("Already scraped: %s", url)
            return None

        session = self._get_worker_session()
        logger.info("Scraping Arabic document: %s", url)
        doc = self._scrape_arabic_document(session, url)
        if not doc:
            with self._stat_lock:
                self.stats["docs_failed"] += 1
            self.state.mark_failed(url, "arabic_scrape_failed", save=False)
            return None

        en_url = doc.get("english_url")
        if en_url:
            logger.info("Scraping English version: %s", en_url)
            en_content = self._scrape_english_document(session, en_url)
            if en_content:
                doc["contentEn"] = en_content

        file_path = self._save_document(doc)
        self.state.mark_completed(url, save=False)
        with self._stat_lock:
            self.stats["docs_scraped"] += 1
        logger.info("Saved document: %s", file_path)
        return doc

    def scrape_document(self, url: str) -> Optional[Dict[str, Any]]:
        """Sequential wrapper for scraping a single document."""
        url = self.normalize_document_url(url)
        return self._scrape_document_worker(url)

    def generate_page_urls(
        self,
        max_pages: Optional[int] = None,
        category_url: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ) -> List[str]:
        """Generate pagination URLs for qanoon.om homepage or a category.

        Args:
            max_pages: Maximum number of listing pages to generate. If None,
                auto-detects from homepage pagination.
            category_url: Optional category URL (e.g., /p/category/.../). If None, uses homepage.

        Returns:
            List of listing page URLs.
        """
        base = (category_url or self.base_url).rstrip("/")
        total = max_pages or self.get_total_pages(category_url, session=session)
        if total <= 0:
            total = 1

        urls = [base + "/"]
        for page_num in range(2, total + 1):
            urls.append(f"{base}/page/{page_num}/")

        logger.info("Generated %d listing page URLs.", len(urls))
        return urls

    def _log_progress(self, total_pending: int) -> None:
        """Log scraping progress and estimated time remaining."""
        elapsed = time.time() - self.stats["started_at"]
        with self._stat_lock:
            done = self.stats["docs_scraped"]
            failed = self.stats["docs_failed"]
        processed = done + failed

        if processed == 0 or elapsed == 0:
            logger.info("Progress: %d scraped, %d failed (%.1fs elapsed)", done, failed, elapsed)
            return

        rate = processed / elapsed
        remaining = total_pending - done
        eta_seconds = remaining / rate if rate > 0 else 0

        logger.info(
            "Progress: %d/%d scraped, %d failed, %.2f docs/s, ETA %.1f min",
            done,
            total_pending,
            failed,
            rate,
            eta_seconds / 60,
        )

    def run(
        self,
        seed_urls: Optional[List[str]] = None,
        max_pages: Optional[int] = None,
        category_url: Optional[str] = None,
        all_docs: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run the scraper.

        Args:
            seed_urls: URLs to discover documents from. If provided, overrides pagination.
            max_pages: Number of listing pages to crawl when seed_urls is not provided.
            category_url: Optional category URL to crawl (e.g., /p/category/.../). Uses homepage if None.
            all_docs: If True, ignore ``max_documents`` and scrape every discovered document.

        Returns:
            List of scraped documents.
        """
        self.stats["started_at"] = time.time()
        documents: List[Dict[str, Any]] = []

        try:
            existing_pending = self.state.get_pending()

            if seed_urls:
                seeds = seed_urls
            elif existing_pending and (all_docs or not self.max_documents or len(existing_pending) >= (self.max_documents or 0)):
                # Resume from existing state: skip re-discovering listing pages
                logger.info(
                    "Resuming from existing state: %d pending documents already discovered. Skipping listing crawl.",
                    len(existing_pending),
                )
                seeds = []
            else:
                pages = None if all_docs else max_pages
                seeds = self.generate_page_urls(max_pages=pages, category_url=category_url)

            # Discover document links sequentially (lightweight listing pages)
            if seeds:
                all_links: List[str] = []
                for idx, seed in enumerate(seeds, 1):
                    links = self.discover_document_links(seed)
                    all_links.extend(links)
                    self.state.add_discovered(links)
                    logger.info("Listing page %d/%d complete.", idx, len(seeds))

                    if not all_docs and self.max_documents:
                        if len(self.state.get_pending()) >= self.max_documents * 2:
                            logger.info("Enough pending documents discovered. Stopping pagination.")
                            break

            pending = self.state.get_pending()
            self.stats["pages_discovered"] = len(pending)

            if self.max_documents and not all_docs:
                pending = pending[: self.max_documents]
                logger.info("Total pending documents: %d (capped at %d)", len(pending), self.max_documents)
            else:
                logger.info("Total pending documents: %d", len(pending))

            # Scrape documents concurrently
            if self.max_workers > 1:
                logger.info("Starting concurrent scraping with %d workers...", self.max_workers)
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_url = {executor.submit(self._scrape_document_worker, url): url for url in pending}
                    for future in as_completed(future_to_url):
                        url = future_to_url[future]
                        try:
                            doc = future.result()
                            if doc:
                                documents.append(doc)
                        except Exception as exc:
                            logger.error("Worker failed for %s: %s", url, exc)
                            with self._stat_lock:
                                self.stats["docs_failed"] += 1
                            self.state.mark_failed(url, str(exc), save=False)

                        processed = self.stats["docs_scraped"] + self.stats["docs_failed"]
                        if processed % 10 == 0:
                            self._log_progress(len(pending))
                            self.state.save()
            else:
                for url in pending:
                    doc = self.scrape_document(url)
                    if doc:
                        documents.append(doc)

                    if (self.stats["docs_scraped"] + self.stats["docs_failed"]) % 10 == 0:
                        self._log_progress(len(pending))
                        self.state.save()

        finally:
            self.stop_playwright()
            self.state.save()
            self.stats["ended_at"] = time.time()
            self._print_summary()

        return documents

    def _print_summary(self) -> None:
        """Print final timing and coverage summary."""
        elapsed = (self.stats["ended_at"] or time.time()) - (self.stats["started_at"] or time.time())
        logger.info("=" * 60)
        logger.info("Scraping summary")
        logger.info("  Elapsed time: %.2f seconds (%.2f minutes)", elapsed, elapsed / 60)
        logger.info("  Documents scraped: %d", self.stats["docs_scraped"])
        logger.info("  Documents failed: %d", self.stats["docs_failed"])
        logger.info("  Total HTTP requests: %d", self.stats["requests_total"])
        logger.info("  Failed HTTP requests: %d", self.stats["requests_failed"])
        if self.stats["docs_scraped"] > 0 and elapsed > 0:
            logger.info("  Average rate: %.3f docs/second", self.stats["docs_scraped"] / elapsed)
        logger.info("=" * 60)


def main() -> None:
    """CLI entrypoint for qanoon.om scraping."""
    import sys

    logging.basicConfig(level=settings.LOG_LEVEL)

    seed_urls = sys.argv[1:] or None
    scraper = QanoonScraper(max_documents=10)
    docs = scraper.run(seed_urls=seed_urls)
    logger.info("Scraped %d documents from qanoon.om", len(docs))


if __name__ == "__main__":
    main()
