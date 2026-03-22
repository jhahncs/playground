"""Springer scraper implementation."""

from typing import List, Optional
from datetime import datetime
from bs4 import BeautifulSoup
import re

from scrapers.base_scraper import BaseScraper
from models.conference import Conference
from utils.logger import get_logger

logger = get_logger(__name__)


class SpringerScraper(BaseScraper):
    """Scraper for Springer conference listings."""

    BASE_URL = "https://www.springer.com"
    CONFERENCES_URL = "https://www.springer.com/gp/computer-science/lncs/conferences"

    def __init__(self, **kwargs):
        """Initialize Springer scraper."""
        super().__init__(**kwargs)

    def parse(self, html: str) -> List[Conference]:
        """Parse HTML into Conference objects."""
        soup = BeautifulSoup(html, 'html.parser')
        conferences = []

        # Springer conference listings
        conference_items = soup.find_all(['div', 'article'], class_=re.compile(r'conference|event'))

        for item in conference_items:
            try:
                conference = self._parse_conference_item(item)
                if conference:
                    conferences.append(conference)
            except Exception as e:
                logger.error(f"Error parsing Springer conference item: {e}")

        return conferences

    def _parse_conference_item(self, item) -> Optional[Conference]:
        """Parse individual conference item."""
        try:
            # Extract title
            title_elem = item.find(['h2', 'h3', 'h4', 'a'], class_=re.compile(r'title|heading'))
            if not title_elem:
                return None

            name = title_elem.get_text(strip=True)

            # Extract URL
            url_elem = item.find('a', href=True)
            website_url = url_elem.get('href') if url_elem else None
            if website_url and not website_url.startswith('http'):
                website_url = self.BASE_URL + website_url

            # Extract dates
            date_elem = item.find(text=re.compile(r'\d{1,2}[/-]\d{1,2}[/-]\d{4}|\w+\s+\d{1,2},\s+\d{4}'))
            conference_start = None
            if date_elem:
                date_text = date_elem.strip()
                conference_start = self._parse_date(date_text)

            # Extract location
            location_elem = item.find(text=re.compile(r'Location:|City:'))
            location = "TBD"
            if location_elem:
                location_text = location_elem.find_next(text=True)
                if location_text:
                    location = location_text.strip()

            # Create conference object
            conference = Conference(
                name=name,
                location=location,
                conference_start=conference_start,
                website_url=website_url,
                source="springer"
            )

            if conference_start:
                conference.year = conference_start.year

            return conference

        except Exception as e:
            logger.error(f"Error parsing Springer conference: {e}")
            return None

    def _parse_date(self, date_str: str):
        """Parse date string to date object."""
        try:
            # Try common date formats
            for fmt in ['%d/%m/%Y', '%m/%d/%Y', '%B %d, %Y', '%d %B %Y', '%Y-%m-%d']:
                try:
                    return datetime.strptime(date_str, fmt).date()
                except ValueError:
                    continue
        except Exception:
            pass
        return None

    def scrape_upcoming_conferences(self, limit: int = 50) -> List[Conference]:
        """
        Scrape upcoming Springer conferences.

        Args:
            limit: Maximum number of conferences

        Returns:
            List of Conference objects
        """
        logger.info(f"Scraping Springer upcoming conferences (limit: {limit})")

        url = self.CONFERENCES_URL

        html = self.fetch_url(url)
        if not html:
            return []

        conferences = self.parse(html)

        logger.info(f"Scraped {len(conferences)} Springer conferences")
        return conferences[:limit]
