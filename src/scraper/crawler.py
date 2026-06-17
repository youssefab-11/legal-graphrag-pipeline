"""Playwright-based web crawler for qanoon.om.

Implements anti-bot evasion, randomized request pacing, exponential backoff,
and resumable checkpointing. Designed to discover and extract both Arabic and
English versions of Omani legal documents.
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext, Response

from src.config.settings import settings
from src.scraper.parser import LegalDocumentParser
from src.scraper.state_manager import StateManager

logger = logging.getLogger(__name__)


class QanoonCrawler:
    """Crawler for the Omani legislation portal https://qanoon.om/"""

    def __init__(
        self,
        base_url: str = settings.QANOON_BASE_URL,
        output_dir: Optional[Path] = None,
        state_manager: Optional[StateManager] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.output_dir = output_dir or settings.RAW_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state = state_manager or StateManager()
        self.parser = LegalDocumentParser()

        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    def _random_delay(self) -> None:
        """Sleep for a randomized interval to mimic human browsing."""
        delay = random.uniform(settings.REQUEST_DELAY_MIN, settings.REQUEST_DELAY_MAX)
        logger.debug("Sleeping %.2fs", delay)
        time.sleep(delay)

    def _get_headers(self) -> Dict[str, str]:
        """Return realistic browser headers."""
        return {
            "User-Agent": settings.USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

    def start(self, headless: bool = True) -> "QanoonCrawler":
        """Initialize Playwright browser and context."""
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=headless)
        self.context = self.browser.new_context(
            user_agent=settings.USER_AGENT,
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
        )
        self.context.set_extra_http_headers(self._get_headers())
        self.page = self.context.new_page()
        logger.info("Browser started.")
        return self

    def stop(self) -> None:
        """Clean up Playwright resources."""
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
        logger.info("Browser stopped.")

    def fetch_page(self, url: str, retries: int = settings.MAX_RETRIES) -> Optional[str]:
        """Fetch a page with retries and anti-bot delays.

        Args:
            url: URL to fetch.
            retries: Number of retry attempts.

        Returns:
            Page HTML content or None if failed.
        """
        if not self.page:
            raise RuntimeError("Crawler not started. Call start() first.")

        for attempt in range(1, retries + 1):
            try:
                self._random_delay()
                response: Response = self.page.goto(url, wait_until="networkidle", timeout=60000)

                if response is None or response.status >= 400:
                    logger.warning("Attempt %d: HTTP %s for %s", attempt, response.status if response else "?", url)
                    if response and response.status == 403:
                        logger.warning("Possible block detected. Increasing delay...")
                        time.sleep(random.uniform(5, 10))
                    continue

                html = self.page.content()
                logger.info("Fetched: %s", url)
                return html

            except Exception as exc:
                logger.warning("Attempt %d failed for %s: %s", attempt, url, exc)
                time.sleep(random.uniform(2, 5) * attempt)

        logger.error("All retries failed for %s", url)
        return None

    def discover_document_links(self, listing_url: str) -> List[str]:
        """Discover document detail page URLs from a listing/category page.

        Args:
            listing_url: URL of a listing page.

        Returns:
            List of absolute document detail URLs.
        """
        html = self.fetch_page(listing_url)
        if not html:
            return []

        soup = self.parser._create_soup(html)
        links: List[str] = []

        # Generic heuristic: links containing law/decree identifiers
        for a in soup.find_all("a", href=True):
            href = a["href"]
            absolute = urljoin(self.base_url, href)
            parsed = urlparse(absolute)

            # Heuristic filters — adjust after inspecting qanoon.om structure
            path = parsed.path.lower()
            if any(kw in path for kw in ["/law/", "/decree/", "/decision/", "/legislation/"]):
                if absolute.startswith(self.base_url):
                    links.append(absolute)

        unique_links = list(set(links))
        logger.info("Discovered %d document links from %s", len(unique_links), listing_url)
        return unique_links

    def extract_metadata(self, soup: Any) -> Dict[str, Any]:
        """Extract metadata from a document detail page.

        This is a generic placeholder. Inspect qanoon.om DOM and override selectors.
        """
        metadata: Dict[str, Any] = {
            "title": "",
            "document_type": "",
            "number": "",
            "issue_date": "",
            "issuer": "",
        }

        # Try common selectors
        title_tag = soup.find("h1") or soup.find(class_=re.compile("title", re.I))
        if title_tag:
            metadata["title"] = title_tag.get_text(strip=True)

        return metadata

    def scrape_document(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape a single document detail page.

        Args:
            url: Document detail URL.

        Returns:
            Parsed document dictionary or None.
        """
        if self.state.is_completed(url):
            logger.info("Already scraped: %s", url)
            return None

        html = self.fetch_page(url)
        if not html:
            self.state.mark_failed(url, "fetch_failed")
            return None

        try:
            soup = self.parser.create_soup(html)
            metadata = self.extract_metadata(soup)
            metadata["id"] = url
            metadata["source_url"] = url
            metadata["language"] = "en"  # Default; language detection refined below

            doc = self.parser.parse_document(html, url, metadata)

            # Save raw HTML for inspection / debugging
            doc_id = doc["id"].replace("https://", "").replace("/", "_")
            raw_path = self.output_dir / f"{doc_id}.html"
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(html)

            # Save parsed JSON
            json_path = self.output_dir / f"{doc_id}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)

            self.state.mark_completed(url)
            logger.info("Scraped and saved document: %s", url)
            return doc

        except Exception as exc:
            logger.error("Parsing failed for %s: %s", url, exc)
            self.state.mark_failed(url, f"parse_failed: {exc}")
            return None

    def run(
        self,
        seed_urls: Optional[List[str]] = None,
        max_documents: int = 100,
    ) -> List[Dict[str, Any]]:
        """Run the crawler.

        Args:
            seed_urls: Starting URLs for discovery.
            max_documents: Maximum number of new documents to scrape.

        Returns:
            List of parsed documents.
        """
        self.start()
        documents: List[Dict[str, Any]] = []

        try:
            seed_urls = seed_urls or [self.base_url]

            # Discover document links from seed pages
            for seed in seed_urls:
                links = self.discover_document_links(seed)
                self.state.add_discovered(links)

            pending = self.state.get_pending()
            logger.info("Starting to scrape %d pending documents (max %d).", len(pending), max_documents)

            for url in pending[:max_documents]:
                doc = self.scrape_document(url)
                if doc:
                    documents.append(doc)

                # Save checkpoint every 5 documents
                if len(documents) % 5 == 0:
                    self.state.save()

        finally:
            self.stop()
            self.state.save()

        logger.info("Crawl complete. Scraped %d new documents.", len(documents))
        return documents


def main() -> None:
    """CLI entrypoint for the crawler."""
    import sys

    logging.basicConfig(level=settings.LOG_LEVEL)

    seed_urls: Optional[List[str]] = sys.argv[1:] or None
    crawler = QanoonCrawler()
    docs = crawler.run(seed_urls=seed_urls, max_documents=50)
    logger.info("Total documents collected: %d", len(docs))


if __name__ == "__main__":
    main()
