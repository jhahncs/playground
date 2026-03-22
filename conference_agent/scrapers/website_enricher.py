"""Website enricher to extract additional information from official conference websites."""

import re
from typing import Optional, Dict, Any
from bs4 import BeautifulSoup
import requests
from utils.logger import get_logger

logger = get_logger(__name__)


class WebsiteEnricher:
    """Extract additional information from conference official websites."""

    def __init__(self, timeout: int = 10, user_agent: Optional[str] = None):
        """
        Initialize website enricher.

        Args:
            timeout: Request timeout in seconds
            user_agent: Custom user agent string
        """
        self.timeout = timeout
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})

    def enrich_from_website(self, url: str) -> Dict[str, Any]:
        """
        Extract additional information from conference website.

        Args:
            url: Conference website URL

        Returns:
            Dictionary with enriched information
        """
        enriched_data = {
            'registration_info': None,
            'keynote_speakers': [],
            'program_committee': [],
            'submission_tracks': [],
            'contact_info': None,
            'social_media': {},
        }

        try:
            logger.info(f"Enriching from website: {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract registration information
            enriched_data['registration_info'] = self._extract_registration_info(soup)

            # Extract keynote speakers
            enriched_data['keynote_speakers'] = self._extract_keynote_speakers(soup)

            # Extract program committee
            enriched_data['program_committee'] = self._extract_program_committee(soup)

            # Extract submission tracks
            enriched_data['submission_tracks'] = self._extract_submission_tracks(soup)

            # Extract contact information
            enriched_data['contact_info'] = self._extract_contact_info(soup)

            # Extract social media links
            enriched_data['social_media'] = self._extract_social_media(soup)

            logger.info(f"Successfully enriched data from {url}")

        except requests.RequestException as e:
            logger.warning(f"Failed to fetch website {url}: {e}")
        except Exception as e:
            logger.error(f"Error enriching from website {url}: {e}")

        return enriched_data

    def _extract_registration_info(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract registration information."""
        patterns = [
            r'registration\s+fee',
            r'early\s+bird',
            r'registration\s+deadline',
        ]

        for pattern in patterns:
            for element in soup.find_all(text=re.compile(pattern, re.IGNORECASE)):
                parent = element.find_parent(['p', 'div', 'section'])
                if parent:
                    text = parent.get_text(strip=True)
                    if len(text) < 500:  # Reasonable length
                        return text

        return None

    def _extract_keynote_speakers(self, soup: BeautifulSoup) -> list:
        """Extract keynote speaker names."""
        speakers = []
        patterns = [
            r'keynote\s+speaker',
            r'invited\s+speaker',
            r'plenary\s+speaker',
        ]

        for pattern in patterns:
            for element in soup.find_all(text=re.compile(pattern, re.IGNORECASE)):
                parent = element.find_parent(['div', 'section'])
                if parent:
                    # Look for names in nearby headings or bold text
                    names = parent.find_all(['h3', 'h4', 'strong', 'b'])
                    for name in names:
                        speaker_name = name.get_text(strip=True)
                        if speaker_name and len(speaker_name.split()) >= 2 and len(speaker_name.split()) <= 5:
                            speakers.append(speaker_name)

        return list(set(speakers[:10]))  # Limit to 10 unique speakers

    def _extract_program_committee(self, soup: BeautifulSoup) -> list:
        """Extract program committee members."""
        committee = []
        patterns = [
            r'program\s+committee',
            r'organizing\s+committee',
            r'PC\s+members',
        ]

        for pattern in patterns:
            for element in soup.find_all(text=re.compile(pattern, re.IGNORECASE)):
                parent = element.find_parent(['div', 'section'])
                if parent:
                    # Look for lists of names
                    lists = parent.find_all(['ul', 'ol'])
                    for lst in lists:
                        items = lst.find_all('li')
                        for item in items[:20]:  # Limit to first 20
                            name = item.get_text(strip=True)
                            if name and len(name.split()) >= 2 and len(name.split()) <= 5:
                                committee.append(name)

        return list(set(committee))

    def _extract_submission_tracks(self, soup: BeautifulSoup) -> list:
        """Extract submission tracks or topics."""
        tracks = []
        patterns = [
            r'submission\s+track',
            r'call\s+for\s+papers',
            r'research\s+topics',
            r'areas\s+of\s+interest',
        ]

        for pattern in patterns:
            for element in soup.find_all(text=re.compile(pattern, re.IGNORECASE)):
                parent = element.find_parent(['div', 'section'])
                if parent:
                    lists = parent.find_all(['ul', 'ol'])
                    for lst in lists:
                        items = lst.find_all('li')
                        for item in items[:15]:  # Limit to 15 tracks
                            track = item.get_text(strip=True)
                            if track and len(track) < 200:
                                tracks.append(track)

        return list(set(tracks))

    def _extract_contact_info(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract contact information."""
        # Look for email addresses
        emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', soup.get_text())

        if emails:
            # Prefer conference-specific emails
            conference_emails = [e for e in emails if 'conference' in e.lower() or 'contact' in e.lower()]
            return conference_emails[0] if conference_emails else emails[0]

        return None

    def _extract_social_media(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract social media links."""
        social_media = {}

        # Twitter/X
        twitter_links = soup.find_all('a', href=re.compile(r'(twitter\.com|x\.com)'))
        if twitter_links:
            social_media['twitter'] = twitter_links[0].get('href')

        # LinkedIn
        linkedin_links = soup.find_all('a', href=re.compile(r'linkedin\.com'))
        if linkedin_links:
            social_media['linkedin'] = linkedin_links[0].get('href')

        # Facebook
        facebook_links = soup.find_all('a', href=re.compile(r'facebook\.com'))
        if facebook_links:
            social_media['facebook'] = facebook_links[0].get('href')

        # YouTube
        youtube_links = soup.find_all('a', href=re.compile(r'youtube\.com'))
        if youtube_links:
            social_media['youtube'] = youtube_links[0].get('href')

        return social_media
