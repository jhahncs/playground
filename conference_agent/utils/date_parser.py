"""Date parsing utilities for various formats."""

from datetime import datetime, date
from dateutil import parser as dateutil_parser
from typing import Optional
import re
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_date(date_str: str, return_start_if_range: bool = True) -> Optional[date]:
    """
    Parse date string in various formats to date object.

    Handles formats like:
    - "Jan 15, 2026"
    - "2026-01-15"
    - "15 Jan 2026"
    - "January 15, 2026"
    - "2026/01/15"
    - "Mar 30, 2026 - Mar 31, 2026" (date ranges)

    Args:
        date_str: Date string to parse
        return_start_if_range: If True, return start date for ranges; if False, return end date

    Returns:
        date object or None if parsing fails
    """
    if not date_str or not isinstance(date_str, str):
        return None

    # Clean the string
    date_str = date_str.strip()

    # Check if it's a date range (contains " - ")
    if ' - ' in date_str or ' to ' in date_str.lower():
        start, end = parse_date_range(date_str)
        return start if return_start_if_range else end

    # Common patterns to handle
    patterns = [
        # ISO format: 2026-01-15
        (r'^\d{4}-\d{2}-\d{2}$', '%Y-%m-%d'),
        # US format: 01/15/2026 or 01-15-2026
        (r'^\d{1,2}[/-]\d{1,2}[/-]\d{4}$', '%m/%d/%Y'),
        # European format: 15/01/2026 or 15-01-2026
        (r'^\d{1,2}[/-]\d{1,2}[/-]\d{4}$', '%d/%m/%Y'),
        # Year first: 2026/01/15
        (r'^\d{4}[/-]\d{1,2}[/-]\d{1,2}$', '%Y/%m/%d'),
    ]

    for pattern, fmt in patterns:
        if re.match(pattern, date_str):
            try:
                return datetime.strptime(date_str.replace('-', '/'), fmt.replace('-', '/')).date()
            except ValueError:
                continue

    # Try dateutil parser as fallback (handles "Jan 15, 2026", etc.)
    try:
        parsed = dateutil_parser.parse(date_str, fuzzy=True)
        return parsed.date()
    except (ValueError, TypeError, dateutil_parser.ParserError) as e:
        logger.warning(f"Failed to parse date: '{date_str}' - {e}")
        return None


def parse_date_range(date_range_str: str) -> tuple[Optional[date], Optional[date]]:
    """
    Parse date range string like "Jan 15-17, 2026" or "Jan 15 - Jan 17, 2026".

    Args:
        date_range_str: Date range string

    Returns:
        Tuple of (start_date, end_date)
    """
    if not date_range_str:
        return None, None

    date_range_str = date_range_str.strip()

    # Pattern: "Jan 15-17, 2026" (same month, different days)
    match = re.match(r'(\w+)\s+(\d+)\s*-\s*(\d+),?\s*(\d{4})', date_range_str)
    if match:
        month, day1, day2, year = match.groups()
        start_str = f"{month} {day1}, {year}"
        end_str = f"{month} {day2}, {year}"
        # Use fuzzy parsing to avoid circular calls
        try:
            start = dateutil_parser.parse(start_str, fuzzy=True).date()
            end = dateutil_parser.parse(end_str, fuzzy=True).date()
            return start, end
        except:
            pass

    # Pattern: "Jan 15 - Jan 17, 2026" or "Jan 15, 2026 - Jan 17, 2026"
    # Split by " - " or " to "
    for separator in [' - ', ' to ', ' TO ']:
        if separator in date_range_str:
            parts = date_range_str.split(separator)
            if len(parts) == 2:
                # Use fuzzy parsing to avoid issues
                try:
                    start = dateutil_parser.parse(parts[0].strip(), fuzzy=True).date()
                    end = dateutil_parser.parse(parts[1].strip(), fuzzy=True).date()
                    return start, end
                except:
                    continue

    # Single date - use fuzzy parsing
    try:
        parsed = dateutil_parser.parse(date_range_str, fuzzy=True).date()
        return parsed, parsed
    except:
        return None, None


def extract_year(text: str) -> Optional[int]:
    """
    Extract 4-digit year from text.

    Args:
        text: Text containing year

    Returns:
        Year as integer or None
    """
    if not text:
        return None

    match = re.search(r'\b(20\d{2})\b', text)
    if match:
        return int(match.group(1))

    return None


def is_past_date(check_date: Optional[date]) -> bool:
    """
    Check if date is in the past.

    Args:
        check_date: Date to check

    Returns:
        True if date is in the past
    """
    if not check_date:
        return False

    return check_date < date.today()


def format_date(date_obj: Optional[date], format_str: str = '%Y-%m-%d') -> str:
    """
    Format date object to string.

    Args:
        date_obj: Date object
        format_str: Format string (default: ISO format)

    Returns:
        Formatted date string or empty string
    """
    if not date_obj:
        return ""

    return date_obj.strftime(format_str)
