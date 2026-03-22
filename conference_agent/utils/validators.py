"""Data validation utilities for conference information."""

from typing import List
from datetime import datetime, date
from models.conference import Conference
from utils.logger import get_logger

logger = get_logger(__name__)


def validate_conference(conference: Conference) -> tuple[bool, List[str]]:
    """
    Validate conference data.

    Args:
        conference: Conference object to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Required fields
    if not conference.name:
        errors.append("Missing conference name")

    if not conference.location:
        errors.append("Missing location")

    if not conference.submission_deadline:
        errors.append("Missing submission deadline")

    if not conference.conference_start:
        errors.append("Missing conference start date")

    # Date logic validation
    if conference.submission_deadline and conference.notification_date:
        if conference.submission_deadline >= conference.notification_date:
            errors.append("Submission deadline must be before notification date")

    if conference.notification_date and conference.conference_start:
        if conference.notification_date >= conference.conference_start:
            errors.append("Notification date must be before conference start")

    if conference.submission_deadline and conference.conference_start:
        if conference.submission_deadline >= conference.conference_start:
            errors.append("Submission deadline must be before conference start")

    if conference.conference_start and conference.conference_end:
        if conference.conference_start > conference.conference_end:
            errors.append("Conference start must be before or equal to conference end")

    # Year validation
    if conference.conference_start:
        if conference.year and conference.year != conference.conference_start.year:
            logger.warning(f"Year mismatch: {conference.year} vs {conference.conference_start.year}")

    # URL validation (basic)
    for url_field, url_value in [
        ('website_url', conference.website_url),
        ('cfp_url', conference.cfp_url)
    ]:
        if url_value and not (url_value.startswith('http://') or url_value.startswith('https://')):
            errors.append(f"Invalid {url_field}: must start with http:// or https://")

    return len(errors) == 0, errors


def calculate_quality_score(conference: Conference) -> float:
    """
    Calculate data quality score (0-1) based on completeness.

    Scoring:
    - Has all required fields (name, submission_deadline, conference_start, location): 0.5
    - Has notification_date: +0.1
    - Has camera_ready_deadline: +0.1
    - Has CFP text: +0.2
    - Has topics/categories: +0.1
    - Has conference_end: +0.05
    - Has website_url: +0.05

    Args:
        conference: Conference object

    Returns:
        Quality score from 0.0 to 1.0
    """
    score = 0.0

    # Required fields (0.5 total)
    has_required = (
        bool(conference.name) and
        bool(conference.submission_deadline) and
        bool(conference.conference_start) and
        bool(conference.location)
    )
    if has_required:
        score += 0.5

    # Optional fields
    if conference.notification_date:
        score += 0.1

    if conference.camera_ready_deadline:
        score += 0.1

    if conference.cfp_text:
        score += 0.2

    if conference.topics or conference.categories:
        score += 0.1

    if conference.conference_end:
        score += 0.05

    if conference.website_url:
        score += 0.05

    return min(score, 1.0)


def filter_valid_conferences(conferences: List[Conference], min_quality: float = 0.5) -> tuple[List[Conference], List[Conference]]:
    """
    Filter conferences by validity and quality score.

    Args:
        conferences: List of conferences to filter
        min_quality: Minimum quality score (default: 0.5)

    Returns:
        Tuple of (valid_conferences, invalid_conferences)
    """
    valid = []
    invalid = []

    for conf in conferences:
        # Calculate quality score
        conf.data_quality_score = calculate_quality_score(conf)

        # Validate
        is_valid, errors = validate_conference(conf)

        if is_valid and conf.data_quality_score >= min_quality:
            valid.append(conf)
        else:
            if errors:
                logger.warning(f"Conference '{conf.name}' validation failed: {', '.join(errors)}")
            if conf.data_quality_score < min_quality:
                logger.warning(f"Conference '{conf.name}' quality score too low: {conf.data_quality_score:.2f}")
            invalid.append(conf)

    return valid, invalid


def filter_future_conferences(conferences: List[Conference]) -> tuple[List[Conference], List[Conference]]:
    """
    Filter conferences to only include those happening this year or later.

    Args:
        conferences: List of conferences to filter

    Returns:
        Tuple of (future_conferences, past_conferences)
    """
    current_year = datetime.now().year
    future = []
    past = []

    for conf in conferences:
        if conf.conference_start and conf.conference_start.year >= current_year:
            future.append(conf)
        else:
            past.append(conf)
            logger.debug(f"Filtering out past conference: {conf.name} ({conf.year})")

    logger.info(f"Filtered: {len(future)} future conferences, {len(past)} past conferences")
    return future, past


def filter_conferences_by_date_range(
    conferences: List[Conference],
    start_date: date = None,
    end_date: date = None
) -> tuple[List[Conference], List[Conference]]:
    """
    Filter conferences by date range based on conference start date.

    Args:
        conferences: List of conferences to filter
        start_date: Minimum conference start date (inclusive). If None, no minimum limit.
        end_date: Maximum conference start date (inclusive). If None, no maximum limit.

    Returns:
        Tuple of (matching_conferences, filtered_out_conferences)
    """
    matching = []
    filtered = []

    for conf in conferences:
        if not conf.conference_start:
            # If no start date, exclude from results
            filtered.append(conf)
            logger.debug(f"Filtering out conference without start date: {conf.name}")
            continue

        include = True
        conf_start = conf.conference_start.date() if isinstance(conf.conference_start, datetime) else conf.conference_start

        # Check start date
        if start_date and conf_start < start_date:
            include = False
            logger.debug(f"Filtering out conference before start date: {conf.name} ({conf.conference_start})")

        # Check end date
        if end_date and conf_start > end_date:
            include = False
            logger.debug(f"Filtering out conference after end date: {conf.name} ({conf.conference_start})")

        if include:
            matching.append(conf)
        else:
            filtered.append(conf)

    logger.info(f"Date range filter: {len(matching)} conferences match, {len(filtered)} filtered out")
    return matching, filtered


def filter_conferences_by_multiple_date_ranges(
    conferences: List[Conference],
    date_ranges: List[tuple[date, date]]
) -> tuple[List[Conference], List[Conference]]:
    """
    Filter conferences by multiple date ranges based on conference start date.
    A conference is included if it falls within ANY of the specified ranges.

    Args:
        conferences: List of conferences to filter
        date_ranges: List of tuples (start_date, end_date). Each range is inclusive.

    Returns:
        Tuple of (matching_conferences, filtered_out_conferences)
    """
    if not date_ranges:
        return conferences, []

    matching = []
    filtered = []

    for conf in conferences:
        if not conf.conference_start:
            # If no start date, exclude from results
            filtered.append(conf)
            logger.debug(f"Filtering out conference without start date: {conf.name}")
            continue

        include = False
        conf_start = conf.conference_start.date() if isinstance(conf.conference_start, datetime) else conf.conference_start

        # Check if conference falls within ANY of the date ranges
        for start_date, end_date in date_ranges:
            if start_date <= conf_start <= end_date:
                include = True
                logger.debug(f"Conference {conf.name} ({conf.conference_start}) matches range {start_date} to {end_date}")
                break

        if include:
            matching.append(conf)
        else:
            filtered.append(conf)
            logger.debug(f"Filtering out conference outside all ranges: {conf.name} ({conf.conference_start})")

    logger.info(f"Multiple date ranges filter: {len(matching)} conferences match, {len(filtered)} filtered out")
    return matching, filtered


def deduplicate_conferences(conferences: List[Conference]) -> List[Conference]:
    """
    Remove duplicate conferences based on (name, year, location).

    Keeps the conference with the higher quality score.

    Args:
        conferences: List of conferences

    Returns:
        Deduplicated list of conferences
    """
    seen = {}

    for conf in conferences:
        # Create key from name, year, and location
        key = (
            conf.name.lower().strip(),
            conf.year,
            conf.location.lower().strip()
        )

        if key not in seen:
            seen[key] = conf
        else:
            # Keep the one with higher quality score
            if conf.data_quality_score > seen[key].data_quality_score:
                logger.info(f"Replacing duplicate conference with higher quality: {conf.name}")
                seen[key] = conf
            else:
                logger.info(f"Skipping duplicate conference: {conf.name}")

    return list(seen.values())
