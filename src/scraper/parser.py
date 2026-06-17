"""HTML / PDF to Markdown transformer.

Converts raw legal document HTML into clean, hierarchical Markdown.
Preserves legal structure (titles, sections, articles, tables) while
stripping advertisements, navigation, scripts, and stylistic markup.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup, Tag
from markdownify import markdownify as md

logger = logging.getLogger(__name__)


class LegalDocumentParser:
    """Parser for converting legal document HTML into structured Markdown."""

    # HTML selectors commonly used for noise elements
    NOISE_SELECTORS = [
        "script",
        "style",
        "nav",
        "header",
        "footer",
        "aside",
        ".advertisement",
        ".ads",
        ".social-share",
        ".comments",
        "#comments",
    ]

    def __init__(self) -> None:
        pass

    @staticmethod
    def create_soup(html: str) -> BeautifulSoup:
        """Create a BeautifulSoup object from HTML string."""
        return BeautifulSoup(html, "lxml")

    @staticmethod
    def clean_html(soup: BeautifulSoup) -> BeautifulSoup:
        """Remove noise elements from parsed HTML."""
        for selector in LegalDocumentParser.NOISE_SELECTORS:
            for element in soup.select(selector):
                element.decompose()
        return soup

    @staticmethod
    def normalize_headings(soup: BeautifulSoup) -> BeautifulSoup:
        """Ensure legal headings map to Markdown heading levels.

        Typical mapping:
        - Document title -> h1 (#)
        - Chapters / Parts -> h2 (##)
        - Sections / Articles -> h3 (###)
        """
        # Attempt to detect headings by class names if semantic tags are missing
        heading_classes = {
            "h1": ["document-title", "law-title", "title"],
            "h2": ["chapter", "part", "book"],
            "h3": ["section", "article", "clause"],
        }

        for tag_name, classes in heading_classes.items():
            for cls in classes:
                for element in soup.find_all(class_=re.compile(cls, re.I)):
                    if element.name not in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                        element.name = tag_name

        return soup

    @staticmethod
    def html_to_markdown(html: str, base_url: Optional[str] = None) -> str:
        """Convert HTML string to clean Markdown.

        Args:
            html: Raw HTML content.
            base_url: Optional base URL for resolving relative links.

        Returns:
            Clean Markdown string.
        """
        soup = BeautifulSoup(html, "lxml")
        soup = LegalDocumentParser.clean_html(soup)
        soup = LegalDocumentParser.normalize_headings(soup)

        # Extract main content area if a clear container exists
        main = soup.find("main") or soup.find("article") or soup.find("div", class_=re.compile("content|document", re.I))
        if main:
            body_html = str(main)
        else:
            body_html = str(soup.find("body") or soup)

        markdown = md(
            body_html,
            heading_style="ATX",
            bullets="-",
            strip=["a"] if base_url is None else None,
        )

        return LegalDocumentParser.post_process(markdown)

    @staticmethod
    def post_process(markdown: str) -> str:
        """Clean up common artifacts from markdown conversion."""
        # Collapse excessive blank lines
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)

        # Remove leading/trailing whitespace per line
        markdown = "\n".join(line.strip() for line in markdown.splitlines())

        # Ensure headings have a space after hashes
        markdown = re.sub(r"^(#{1,6})([^ #])", r"\1 \2", markdown, flags=re.MULTILINE)

        # Remove leftover HTML tags if any
        markdown = re.sub(r"<[^>]+>", "", markdown)

        return markdown.strip()

    def parse_document(
        self,
        html: str,
        source_url: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Parse a single HTML document into a structured dictionary.

        Args:
            html: Raw HTML content.
            source_url: URL the content was fetched from.
            metadata: Optional pre-extracted metadata (title, date, etc.).

        Returns:
            Document dictionary ready for ingestion.
        """
        metadata = metadata or {}
        markdown = self.html_to_markdown(html, base_url=source_url)

        doc: Dict[str, Any] = {
            "id": metadata.get("id") or source_url,
            "source_url": source_url,
            "title": metadata.get("title", ""),
            "document_type": metadata.get("document_type", ""),
            "number": metadata.get("number", ""),
            "issue_date": metadata.get("issue_date", ""),
            "issuer": metadata.get("issuer", ""),
            "raw_markdown": markdown,
        }

        # Placeholder: language detection and contentAr/contentEn assignment
        # will be refined by the crawler when fetching both language versions.
        doc["contentEn"] = markdown if metadata.get("language") == "en" else ""
        doc["contentAr"] = markdown if metadata.get("language") == "ar" else ""

        return doc


class PDFParser:
    """Minimal PDF parser using pypdf for downloadable legal documents."""

    @staticmethod
    def extract_text(pdf_path: str) -> str:
        """Extract text from a PDF file.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Extracted text as string.
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            logger.error("pypdf is not installed. Install it to parse PDFs.")
            return ""

        text_parts: List[str] = []
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        except Exception as exc:
            logger.error("Failed to parse PDF %s: %s", pdf_path, exc)

        return "\n\n".join(text_parts)
