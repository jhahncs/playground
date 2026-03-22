"""Conference data model."""

from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import List, Optional
import uuid


@dataclass
class Conference:
    """Conference information data model."""

    # Identifiers
    conference_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    series_id: Optional[str] = None

    # Basic Information
    name: str = ""
    acronym: Optional[str] = None
    year: int = 0

    # Location
    location: str = ""
    venue: Optional[str] = None
    is_virtual: bool = False
    is_hybrid: bool = False

    # Dates
    conference_start: Optional[date] = None
    conference_end: Optional[date] = None

    # Deadlines
    submission_deadline: Optional[date] = None
    notification_date: Optional[date] = None
    camera_ready_deadline: Optional[date] = None

    # Topics
    topics: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)

    # CFP Details
    cfp_text: Optional[str] = None
    cfp_url: Optional[str] = None
    website_url: Optional[str] = None

    # Metadata
    source: str = "wikicfp"
    last_updated: datetime = field(default_factory=datetime.now)
    data_quality_score: float = 0.0

    def __post_init__(self):
        """Validate data after initialization."""
        # Ensure lists are lists
        if not isinstance(self.topics, list):
            self.topics = []
        if not isinstance(self.categories, list):
            self.categories = []

    def is_valid(self) -> bool:
        """Check if conference has minimum required fields."""
        return bool(
            self.name and
            self.submission_deadline and
            self.conference_start and
            self.location
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV/Excel export."""
        data = asdict(self)
        # Convert dates to strings
        for key in ['conference_start', 'conference_end', 'submission_deadline',
                   'notification_date', 'camera_ready_deadline']:
            if data[key]:
                data[key] = data[key].isoformat() if isinstance(data[key], date) else data[key]

        # Convert datetime to string
        if data['last_updated']:
            data['last_updated'] = data['last_updated'].isoformat()

        # Convert lists to comma-separated strings
        data['topics'] = ', '.join(data['topics']) if data['topics'] else ''
        data['categories'] = ', '.join(data['categories']) if data['categories'] else ''

        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'Conference':
        """Create Conference from dictionary."""
        # Parse dates
        for key in ['conference_start', 'conference_end', 'submission_deadline',
                   'notification_date', 'camera_ready_deadline']:
            if data.get(key):
                if isinstance(data[key], str):
                    data[key] = datetime.fromisoformat(data[key]).date()

        # Parse datetime
        if data.get('last_updated'):
            if isinstance(data['last_updated'], str):
                data['last_updated'] = datetime.fromisoformat(data['last_updated'])

        # Parse lists
        if data.get('topics') and isinstance(data['topics'], str):
            data['topics'] = [t.strip() for t in data['topics'].split(',') if t.strip()]
        if data.get('categories') and isinstance(data['categories'], str):
            data['categories'] = [c.strip() for c in data['categories'].split(',') if c.strip()]

        return cls(**data)
