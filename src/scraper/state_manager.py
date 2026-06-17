"""Crawl state manager for resumable scraping.

Persistently tracks discovered URLs, completed URLs, and failed URLs so that
crawling can resume after interruption without losing progress.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.config.settings import settings

logger = logging.getLogger(__name__)


class StateManager:
    """Manages checkpoint state for the web crawler."""

    DEFAULT_FILENAME = "crawl_state.json"

    def __init__(self, checkpoint_dir: Optional[Path] = None, filename: Optional[str] = None) -> None:
        self.checkpoint_dir = checkpoint_dir or settings.CHECKPOINT_DIR
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.checkpoint_dir / (filename or self.DEFAULT_FILENAME)

        self.discovered: Set[str] = set()
        self.completed: Set[str] = set()
        self.failed: Dict[str, str] = {}
        self.stats: Dict[str, Any] = {
            "started_at": None,
            "last_updated": None,
            "documents_scraped": 0,
        }

        self.load()

    def load(self) -> None:
        """Load state from disk if it exists."""
        if not self.state_file.exists():
            logger.info("No existing checkpoint found at %s. Starting fresh.", self.state_file)
            return

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.discovered = set(data.get("discovered", []))
            self.completed = set(data.get("completed", []))
            self.failed = {k: v for k, v in data.get("failed", {}).items()}
            self.stats = data.get("stats", self.stats)

            logger.info(
                "Checkpoint loaded: %d discovered, %d completed, %d failed.",
                len(self.discovered),
                len(self.completed),
                len(self.failed),
            )
        except (json.JSONDecodeError, IOError) as exc:
            logger.error("Failed to load checkpoint: %s. Starting fresh.", exc)

    def save(self) -> None:
        """Persist current state to disk."""
        self.stats["last_updated"] = datetime.utcnow().isoformat()

        data = {
            "discovered": sorted(self.discovered),
            "completed": sorted(self.completed),
            "failed": self.failed,
            "stats": self.stats,
        }

        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug("Checkpoint saved to %s", self.state_file)
        except IOError as exc:
            logger.error("Failed to save checkpoint: %s", exc)

    def add_discovered(self, urls: List[str]) -> None:
        """Add URLs to the discovered set."""
        before = len(self.discovered)
        self.discovered.update(urls)
        if len(self.discovered) > before:
            self.save()

    def mark_completed(self, url: str) -> None:
        """Mark a URL as successfully scraped."""
        self.completed.add(url)
        self.discarded(url)
        self.stats["documents_scraped"] = self.stats.get("documents_scraped", 0) + 1
        self.save()

    def mark_failed(self, url: str, reason: str) -> None:
        """Mark a URL as failed with a reason."""
        self.failed[url] = reason
        self.discarded(url)
        self.save()

    def discarded(self, url: str) -> None:
        """No-op placeholder for future priority queue logic."""
        pass

    def get_pending(self) -> List[str]:
        """Return URLs that have been discovered but not completed or failed."""
        return sorted(self.discovered - self.completed - set(self.failed.keys()))

    def is_completed(self, url: str) -> bool:
        """Check if a URL has already been successfully scraped."""
        return url in self.completed

    def reset_failed(self) -> None:
        """Clear failed URLs so they can be retried."""
        count = len(self.failed)
        self.failed.clear()
        self.save()
        logger.info("Reset %d failed URLs for retry.", count)

    def reset_all(self) -> None:
        """Reset all state. Use with caution."""
        self.discovered.clear()
        self.completed.clear()
        self.failed.clear()
        self.stats = {
            "started_at": None,
            "last_updated": None,
            "documents_scraped": 0,
        }
        self.save()
        logger.warning("All crawl state reset.")
