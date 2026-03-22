"""WikiCFP scraper implementation."""

from typing import List, Optional
from scrapers.base_scraper import BaseScraper
from parsers.wikicfp_parser import WikiCFPParser
from models.conference import Conference
from utils.logger import get_logger
from tqdm import tqdm

logger = get_logger(__name__)


class WikiCFPScraper(BaseScraper):
    """Scraper for WikiCFP.com"""

    BASE_URL = "http://www.wikicfp.com/cfp/"

    def __init__(self, **kwargs):
        """Initialize WikiCFP scraper with 5-second rate limit."""
        if 'rate_limit' not in kwargs:
            kwargs['rate_limit'] = 5.0
        super().__init__(**kwargs)
        self.parser = WikiCFPParser()

    def parse(self, html: str) -> List[Conference]:
        """Parse HTML into Conference objects."""
        return [self.parser.parse_conference_detail(html)]

    def scrape_category(self, category: str, max_pages: int = 1) -> List[Conference]:
        """
        Scrape conferences from a specific category.

        Args:
            category: Category name (e.g., "machine learning", "computer vision")
            max_pages: Maximum number of pages to scrape

        Returns:
            List of Conference objects
        """
        logger.info(f"Scraping category: {category} (max {max_pages} pages)")

        conferences = []

        for page in range(1, max_pages + 1):
            # Build category URL
            url = f"{self.BASE_URL}call?conference={category.replace(' ', '+')}"
            if page > 1:
                url += f"&page={page}"

            logger.info(f"Fetching page {page}: {url}")

            html = self.fetch_url(url)
            if not html:
                logger.warning(f"Failed to fetch page {page}")
                break

            # Parse conference list
            conf_list = self.parser.parse_conference_list(html)

            if not conf_list:
                logger.info("No more conferences found")
                break

            logger.info(f"Found {len(conf_list)} conferences on page {page}")

            # Fetch details for each conference
            for conf_info in tqdm(conf_list, desc=f"Page {page}", unit="conference"):
                if conf_info.get('event_id'):
                    conference = self.scrape_conference(conf_info['event_id'])
                    if conference:
                        conferences.append(conference)

        logger.info(f"Scraped {len(conferences)} conferences from category '{category}'")
        return conferences

    def scrape_conference(self, event_id: str) -> Optional[Conference]:
        """
        Scrape individual conference by event ID.

        Args:
            event_id: WikiCFP event ID

        Returns:
            Conference object or None
        """
        url = f"{self.BASE_URL}servlet/event.showcfp?eventid={event_id}"

        logger.debug(f"Fetching conference {event_id}")

        html = self.fetch_url(url)
        if not html:
            return None

        conference = self.parser.parse_conference_detail(html, event_id)

        if conference:
            # Extract acronym if not present
            if not conference.acronym:
                conference.acronym = self.parser.extract_acronym(conference.name)

            # Set CFP URL
            conference.cfp_url = url

        return conference

    def scrape_latest_cfps(self, days_back: int = 30, limit: int = 100) -> List[Conference]:
        """
        Scrape recently posted CFPs.

        Args:
            days_back: Number of days to look back
            limit: Maximum number of conferences to scrape

        Returns:
            List of Conference objects
        """
        logger.info(f"Scraping latest CFPs from last {days_back} days (limit: {limit})")

        url = f"{self.BASE_URL}allcfp"

        html = self.fetch_url(url)
        if not html:
            return []

        conf_list = self.parser.parse_conference_list(html)

        conferences = []

        for conf_info in tqdm(conf_list[:limit], desc="Latest CFPs", unit="conference"):
            if conf_info.get('event_id'):
                conference = self.scrape_conference(conf_info['event_id'])
                if conference:
                    conferences.append(conference)

        logger.info(f"Scraped {len(conferences)} latest conferences")
        return conferences

    def search_by_keyword(self, keyword: str, limit: int = 50) -> List[Conference]:
        """
        Search conferences by keyword.

        Args:
            keyword: Search keyword
            limit: Maximum number of results

        Returns:
            List of Conference objects
        """
        logger.info(f"Searching for keyword: '{keyword}' (limit: {limit})")

        # WikiCFP search URL
        url = f"{self.BASE_URL}call?conference={keyword.replace(' ', '+')}"

        html = self.fetch_url(url)
        if not html:
            return []

        conf_list = self.parser.parse_conference_list(html)

        conferences = []

        for conf_info in tqdm(conf_list[:limit], desc=f"Searching '{keyword}'", unit="conference"):
            if conf_info.get('event_id'):
                conference = self.scrape_conference(conf_info['event_id'])
                if conference:
                    conferences.append(conference)

        logger.info(f"Found {len(conferences)} conferences for keyword '{keyword}'")
        return conferences
