"""Base scraper with rate limiting and retry logic."""

from abc import ABC, abstractmethod
import time
import requests
from typing import Optional, Dict, List
import hashlib
import os
import pickle
from utils.logger import get_logger

logger = get_logger(__name__)


class BaseScraper(ABC):
    """Abstract base class for all scrapers."""

    def __init__(
        self,
        rate_limit: float = 5.0,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        cache_dir: str = "data/cache",
        use_cache: bool = True
    ):
        """
        Initialize base scraper.

        Args:
            rate_limit: Minimum seconds between requests
            max_retries: Maximum number of retry attempts
            retry_backoff: Exponential backoff multiplier
            cache_dir: Directory for caching responses
            use_cache: Whether to use caching
        """
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.last_request_time = 0
        self.session = requests.Session()

        # Set user agent
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # Create cache directory
        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            logger.info(f"Created cache directory: {self.cache_dir}")

    def _get_cache_key(self, url: str) -> str:
        """Generate cache key from URL."""
        return hashlib.md5(url.encode()).hexdigest()

    def _get_from_cache(self, url: str) -> Optional[str]:
        """Get cached response if available."""
        if not self.use_cache:
            return None

        cache_key = self._get_cache_key(url)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    logger.debug(f"Cache hit for URL: {url}")
                    return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cache for {url}: {e}")

        return None

    def _save_to_cache(self, url: str, content: str):
        """Save response to cache."""
        if not self.use_cache:
            return

        cache_key = self._get_cache_key(url)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(content, f)
            logger.debug(f"Cached response for URL: {url}")
        except Exception as e:
            logger.warning(f"Failed to cache response for {url}: {e}")

    def _enforce_rate_limit(self):
        """Ensure minimum time between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit:
            sleep_time = self.rate_limit - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def fetch_url(self, url: str, timeout: int = 30) -> Optional[str]:
        """
        Fetch URL with rate limiting, caching, and retry logic.

        Args:
            url: URL to fetch
            timeout: Request timeout in seconds

        Returns:
            HTML content or None if failed
        """
        # Check cache first
        cached = self._get_from_cache(url)
        if cached:
            return cached

        # Enforce rate limiting
        self._enforce_rate_limit()

        # Retry logic
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Fetching URL (attempt {attempt + 1}/{self.max_retries}): {url}")

                response = self.session.get(url, timeout=timeout)
                response.raise_for_status()

                content = response.text

                # Save to cache
                self._save_to_cache(url, content)

                return content

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout fetching {url} (attempt {attempt + 1})")

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Too Many Requests
                    wait_time = self.retry_backoff ** (attempt + 1)
                    logger.warning(f"Rate limited (429). Waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                elif e.response.status_code == 404:
                    logger.error(f"Page not found (404): {url}")
                    return None
                else:
                    logger.error(f"HTTP error {e.response.status_code}: {url}")

            except requests.exceptions.ConnectionError:
                logger.error(f"Connection error for {url}")

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed for {url}: {e}")

            # Exponential backoff between retries
            if attempt < self.max_retries - 1:
                wait_time = self.retry_backoff ** attempt
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)

        logger.error(f"Failed to fetch {url} after {self.max_retries} attempts")
        return None

    def clear_cache(self):
        """Clear all cached responses."""
        if not os.path.exists(self.cache_dir):
            return

        count = 0
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                os.remove(os.path.join(self.cache_dir, filename))
                count += 1

        logger.info(f"Cleared {count} cached files")

    @abstractmethod
    def parse(self, html: str) -> List:
        """
        Parse HTML into data objects.

        Must be implemented by subclasses.

        Args:
            html: HTML content to parse

        Returns:
            List of parsed objects
        """
        pass
