"""Specialized scraper for qanoon.om and decree.om.

Extracts Omani legal documents in Arabic and English, converts them to
Markdown, and prepares them for ingestion into the GraphRAG pipeline.
"""

import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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

    def __init__(
        self,
        base_url: str = "https://qanoon.om",
        english_base_url: str = "https://decree.om",
        output_dir: Optional[Path] = None,
        state_manager: Optional[StateManager] = None,
        max_documents: int = 20,
        delay_min: float = 2.0,
        delay_max: float = 5.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.english_base_url = english_base_url.rstrip("/")
        self.output_dir = output_dir or settings.RAW_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state = state_manager or StateManager(filename="qanoon_state.json")
        self.max_documents = max_documents
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.parser = LegalDocumentParser()

        self.playwright = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

    def _random_delay(self) -> None:
        """Sleep for a randomized interval."""
        delay = random.uniform(self.delay_min, self.delay_max)
        time.sleep(delay)

    def start(self, headless: bool = True) -> "QanoonScraper":
        """Initialize Playwright browser."""
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=headless)
        self.page = self.browser.new_page(
            viewport={"width": 1920, "height": 1080},
            user_agent=settings.USER_AGENT,
        )
        self.page.set_extra_http_headers({
            "Accept-Language": "ar,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })
        logger.info("Browser started for qanoon.om scraping.")
        return self

    def stop(self) -> None:
        """Clean up browser resources."""
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
        logger.info("Browser stopped.")

    def fetch_page(self, url: str, retries: int = 3) -> Optional[str]:
        """Fetch a page with retries and anti-bot delays."""
        if not self.page:
            raise RuntimeError("Scraper not started. Call start() first.")

        for attempt in range(1, retries + 1):
            try:
                self._random_delay()
                response: Response = self.page.goto(url, wait_until="networkidle", timeout=60000)
                if response is None or response.status >= 400:
                    logger.warning("Attempt %d: HTTP %s for %s", attempt, response.status if response else "?", url)
                    continue
                return self.page.content()
            except Exception as exc:
                logger.warning("Attempt %d failed for %s: %s", attempt, url, exc)
                time.sleep(random.uniform(2, 5) * attempt)

        logger.error("All retries failed for %s", url)
        return None

    def discover_document_links(self, listing_url: str) -> List[str]:
        """Discover document detail URLs from a listing page."""
        html = self.fetch_page(listing_url)
        if not html:
            return []

        soup = self.parser.create_soup(html)
        links: List[str] = []

        for a in soup.find_all("a", href=True):
            href = a["href"]
            absolute = urljoin(self.base_url, href)
            parsed = urlparse(absolute)

            if parsed.netloc == urlparse(self.base_url).netloc and self.DOC_URL_PATTERN.match(parsed.path):
                # Normalize URL
                normalized = absolute.rstrip("/") + "/"
                if normalized not in links:
                    links.append(normalized)

        logger.info("Discovered %d document links from %s", len(links), listing_url)
        return links

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

        # Title
        title_tag = soup.find("h1", class_="entry-title") or soup.find("h2", class_="entry-title")
        if title_tag:
            metadata["title"] = title_tag.get_text(strip=True)

        # Category / document type
        cat_tag = soup.find("div", class_="entry-categories")
        if cat_tag:
            cat_link = cat_tag.find("a")
            if cat_link:
                metadata["document_type"] = cat_link.get_text(strip=True)

        # Date
        date_tag = soup.find("li", class_="post-date")
        if date_tag:
            date_link = date_tag.find("a")
            if date_link:
                metadata["issue_date_text"] = date_link.get_text(strip=True)

        # Try to extract document number from title
        title = metadata.get("title", "")
        number_match = re.search(r"رقم\s*([٠١٢٣٤٥٦٧٨٩\d]+\s*/\s*[٠١٢٣٤٥٦٧٨٩\d]+)", title)
        if number_match:
            metadata["number"] = self.normalize_arabic_number(number_match.group(1).replace(" ", ""))

        # Try to extract issuer from title
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

    def scrape_arabic_document(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape Arabic document page."""
        html = self.fetch_page(url)
        if not html:
            return None

        soup = self.parser.create_soup(html)
        metadata = self.extract_arabic_metadata(soup, url)

        # Extract main content
        content_div = soup.find("div", class_="entry-content")
        if not content_div:
            logger.warning("No entry-content found for %s", url)
            return None

        markdown = self.parser.html_to_markdown(str(content_div), base_url=self.base_url)
        metadata["contentAr"] = markdown
        metadata["raw_markdown"] = markdown
        metadata["english_url"] = self.extract_english_url(soup)
        metadata["pdf_url"] = self.extract_pdf_url(soup)

        # Generate ID from URL
        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")
        metadata["id"] = "-".join(path_parts[-2:]) if len(path_parts) >= 2 else parsed.path.strip("/").replace("/", "-")

        return metadata

    def scrape_english_document(self, en_url: str) -> Optional[str]:
        """Scrape English document page and return markdown content."""
        html = self.fetch_page(en_url)
        if not html:
            return None

        soup = self.parser.create_soup(html)
        content_div = soup.find("div", class_="entry-content")
        if not content_div:
            logger.warning("No entry-content found for English page %s", en_url)
            return None

        return self.parser.html_to_markdown(str(content_div), base_url=self.english_base_url)

    def scrape_document(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape both Arabic and English versions of a document."""
        if self.state.is_completed(url):
            logger.info("Already scraped: %s", url)
            return None

        logger.info("Scraping Arabic document: %s", url)
        doc = self.scrape_arabic_document(url)
        if not doc:
            self.state.mark_failed(url, "arabic_scrape_failed")
            return None

        # Scrape English version if available
        en_url = doc.get("english_url")
        if en_url:
            logger.info("Scraping English version: %s", en_url)
            en_content = self.scrape_english_document(en_url)
            if en_content:
                doc["contentEn"] = en_content

        # Save JSON
        file_name = f"{doc['id']}.json"
        file_path = self.output_dir / file_name
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

        self.state.mark_completed(url)
        logger.info("Saved document: %s", file_path)
        return doc

    def generate_page_urls(self, max_pages: int = 5, category_url: Optional[str] = None) -> List[str]:
        """Generate pagination URLs for qanoon.om homepage or a category.

        Args:
            max_pages: Maximum number of listing pages to generate.
            category_url: Optional category URL (e.g., /p/category/.../). If None, uses homepage.

        Returns:
            List of listing page URLs.
        """
        base = (category_url or self.base_url).rstrip("/")
        urls = [base + "/"]
        for page_num in range(2, max_pages + 1):
            urls.append(f"{base}/page/{page_num}/")
        return urls

    def run(
        self,
        seed_urls: Optional[List[str]] = None,
        max_pages: int = 5,
        category_url: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run the scraper.

        Args:
            seed_urls: URLs to discover documents from. If provided, overrides pagination.
            max_pages: Number of listing pages to crawl when seed_urls is not provided.
            category_url: Optional category URL to crawl (e.g., /p/category/.../). Uses homepage if None.

        Returns:
            List of scraped documents.
        """
        self.start()
        documents: List[Dict[str, Any]] = []

        try:
            if seed_urls:
                seeds = seed_urls
            else:
                seeds = self.generate_page_urls(max_pages=max_pages, category_url=category_url)

            # Discover document links
            all_links: List[str] = []
            for seed in seeds:
                links = self.discover_document_links(seed)
                all_links.extend(links)
                self.state.add_discovered(links)

                # Stop early if we already have enough pending documents
                if len(self.state.get_pending()) >= self.max_documents * 2:
                    logger.info("Enough pending documents discovered. Stopping pagination.")
                    break

            # Deduplicate and limit
            pending = self.state.get_pending()
            logger.info("Total pending documents: %d (max: %d)", len(pending), self.max_documents)

            for url in pending[: self.max_documents]:
                doc = self.scrape_document(url)
                if doc:
                    documents.append(doc)

                if len(documents) % 5 == 0:
                    self.state.save()

        finally:
            self.stop()
            self.state.save()

        logger.info("Qanoon scraping complete. Collected %d documents.", len(documents))
        return documents


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
