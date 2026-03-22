"""Parser for WikiCFP HTML content."""

from bs4 import BeautifulSoup
from typing import List, Optional
from models.conference import Conference
from utils.date_parser import parse_date, parse_date_range, extract_year
from utils.logger import get_logger
from datetime import datetime
import re

logger = get_logger(__name__)


class WikiCFPParser:
    """Parser for WikiCFP.com HTML content."""

    @staticmethod
    def parse_conference_list(html: str) -> List[dict]:
        """
        Parse conference listing page.

        Args:
            html: HTML content from WikiCFP category page

        Returns:
            List of conference dictionaries with basic info
        """
        soup = BeautifulSoup(html, 'html.parser')
        conferences = []

        # WikiCFP uses tables for conference listings
        # Look for tables with class 'sortable' or containing conference data
        tables = soup.find_all('table')

        for table in tables:
            rows = table.find_all('tr')

            for row in rows[1:]:  # Skip header row
                cols = row.find_all('td')

                if len(cols) < 3:
                    continue

                try:
                    # Extract event link and name
                    event_link = cols[0].find('a')
                    if not event_link:
                        continue

                    event_url = event_link.get('href', '')
                    event_name = event_link.get_text(strip=True)

                    # Extract event ID from URL
                    event_id = None
                    id_match = re.search(r'eventid=(\d+)', event_url)
                    if id_match:
                        event_id = id_match.group(1)

                    # Extract deadline (usually in 2nd or 3rd column)
                    deadline_text = cols[1].get_text(strip=True) if len(cols) > 1 else ""

                    # Extract location (usually in last columns)
                    location_text = cols[-1].get_text(strip=True) if len(cols) > 2 else ""

                    # Extract conference dates
                    conf_dates_text = cols[2].get_text(strip=True) if len(cols) > 3 else ""

                    conferences.append({
                        'event_id': event_id,
                        'name': event_name,
                        'event_url': event_url,
                        'deadline_text': deadline_text,
                        'location_text': location_text,
                        'dates_text': conf_dates_text
                    })

                except Exception as e:
                    logger.warning(f"Failed to parse conference row: {e}")
                    continue

        logger.info(f"Parsed {len(conferences)} conferences from listing page")
        return conferences

    @staticmethod
    def parse_conference_detail(html: str, event_id: str = None) -> Optional[Conference]:
        """
        Parse individual conference detail page.

        Args:
            html: HTML content from WikiCFP conference detail page
            event_id: Optional event ID

        Returns:
            Conference object or None
        """
        soup = BeautifulSoup(html, 'html.parser')

        try:
            conference = Conference()

            if event_id:
                conference.conference_id = f"wikicfp_{event_id}"

            # Extract conference name (usually in h2 or first heading)
            name_tag = soup.find('h2') or soup.find('h1')
            if name_tag:
                conference.name = name_tag.get_text(strip=True)

            # Find the main table with conference information
            # WikiCFP uses table with class "gglu" containing th/td pairs
            main_table = soup.find('table', class_='gglu')

            if not main_table:
                # Fallback: look for any table with submission deadline
                tables = soup.find_all('table')
                for table in tables:
                    text = table.get_text()
                    if 'submission' in text.lower() and 'deadline' in text.lower():
                        main_table = table
                        break

            if not main_table:
                logger.warning("Could not find main conference information table")
                return None

            # Parse table rows - WikiCFP uses <th> for labels and <td> for values
            rows = main_table.find_all('tr')

            for row in rows:
                # Look for th (label) and td (value) pairs
                th = row.find('th')
                td = row.find('td')

                if not th or not td:
                    continue

                label = th.get_text(strip=True).lower()
                value = td.get_text(strip=True)

                # Parse different fields
                if 'submission' in label and 'deadline' in label:
                    conference.submission_deadline = parse_date(value)

                elif 'notification' in label:
                    conference.notification_date = parse_date(value)

                elif 'camera' in label and 'ready' in label:
                    conference.camera_ready_deadline = parse_date(value)

                elif 'when' in label or ('conference' in label and 'date' in label):
                    # Parse conference date range
                    start, end = parse_date_range(value)
                    conference.conference_start = start
                    conference.conference_end = end

                elif 'where' in label or 'location' in label:
                    conference.location = value

                    # Check for virtual/hybrid
                    value_lower = value.lower()
                    if 'virtual' in value_lower or 'online' in value_lower:
                        conference.is_virtual = True
                    if 'hybrid' in value_lower:
                        conference.is_hybrid = True

                elif 'categories' in label or 'topics' in label:
                    # Split by comma or semicolon
                    categories = [c.strip() for c in re.split(r'[,;]', value) if c.strip()]
                    conference.categories.extend(categories)

            # Extract year from dates or name
            if conference.conference_start:
                conference.year = conference.conference_start.year
            else:
                year = extract_year(conference.name)
                if year:
                    conference.year = year

            # Extract CFP URL and website URL
            links = soup.find_all('a')
            for link in links:
                href = link.get('href', '')
                text = link.get_text(strip=True).lower()

                if 'link' in text or 'website' in text or 'home' in text:
                    if href.startswith('http'):
                        conference.website_url = href
                        break

            # Extract CFP text from page content
            cfp_section = soup.find(text=re.compile(r'call for papers', re.I))
            if cfp_section:
                parent = cfp_section.find_parent()
                if parent:
                    conference.cfp_text = parent.get_text(strip=True)[:1000]  # Limit length

            # Set metadata
            conference.source = "wikicfp"
            conference.last_updated = datetime.now()

            return conference

        except Exception as e:
            logger.error(f"Failed to parse conference detail: {e}")
            return None

    @staticmethod
    def extract_acronym(name: str) -> Optional[str]:
        """
        Extract conference acronym from name.

        Examples:
        - "CVPR 2026" -> "CVPR"
        - "International Conference on Machine Learning (ICML)" -> "ICML"

        Args:
            name: Conference name

        Returns:
            Acronym or None
        """
        # Pattern: Text in parentheses
        match = re.search(r'\(([A-Z]+\d*)\)', name)
        if match:
            return match.group(1)

        # Pattern: Leading uppercase acronym
        match = re.match(r'^([A-Z]+\d*)\s+\d{4}', name)
        if match:
            return match.group(1)

        # Pattern: Leading uppercase acronym followed by colon or dash
        match = re.match(r'^([A-Z]+\d*)[\s:-]', name)
        if match:
            return match.group(1)

        return None
