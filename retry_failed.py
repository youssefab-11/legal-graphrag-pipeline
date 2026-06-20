"""Retry previously failed scrape URLs without full rediscovery."""
import logging

from src.config.settings import settings
from src.scraper.qanoon_scraper import QanoonScraper
from src.scraper.state_manager import StateManager


def main() -> None:
    logging.basicConfig(level=settings.LOG_LEVEL)

    state = StateManager(filename="qanoon_state.json")
    pending = state.get_pending()
    failed_urls = list(state.failed.keys()) if state.failed else []

    print(f"Pending URLs: {len(pending)}")
    print(f"Failed URLs: {len(failed_urls)}")

    urls_to_retry = pending or failed_urls
    if not urls_to_retry:
        print("No URLs to retry.")
        return

    print(f"Retrying {len(urls_to_retry)} URLs...")

    scraper = QanoonScraper(
        max_documents=len(urls_to_retry),
        delay_min=0.5,
        delay_max=1.5,
        max_workers=1,
    )
    docs = scraper.run(seed_urls=urls_to_retry)

    print(f"Successfully retried {len(docs)} documents.")


if __name__ == "__main__":
    main()
