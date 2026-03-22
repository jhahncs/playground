"""Configuration settings for conference agent."""

import os

# ==========================================
# WikiCFP Settings
# ==========================================

WIKICFP_BASE_URL = "http://www.wikicfp.com/cfp/"
RATE_LIMIT_SECONDS = 5.0  # Minimum seconds between requests
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # Exponential backoff multiplier

# ==========================================
# Categories to Scrape
# ==========================================

# Common academic disciplines
CATEGORIES = [
    "machine learning",
    "artificial intelligence",
    "computer vision",
    "natural language processing",
    "data mining",
    "deep learning",
    "information retrieval",
    "human computer interaction",
    "software engineering",
    "computer networks",
    "cybersecurity",
    "database",
    "distributed systems",
    "cloud computing",
]

# ==========================================
# Data Quality Settings
# ==========================================

MIN_QUALITY_SCORE = 0.5  # Minimum score to include in final output
ENRICH_THRESHOLD = 0.7   # Scrape official website if score below this

# ==========================================
# Storage Paths
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

CSV_OUTPUT_PATH = os.path.join(PROCESSED_DIR, "conferences.csv")
EXCEL_OUTPUT_PATH = os.path.join(PROCESSED_DIR, "conferences.xlsx")
SUMMARY_OUTPUT_PATH = os.path.join(PROCESSED_DIR, "summary.txt")

# ==========================================
# Logging Settings
# ==========================================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = os.path.join(BASE_DIR, "conference_agent.log")

# ==========================================
# Scraping Limits
# ==========================================

MAX_PAGES_PER_CATEGORY = 5  # Maximum pages to scrape per category
MAX_CONFERENCES_PER_RUN = 200  # Safety limit to avoid excessive scraping

# ==========================================
# Feature Flags
# ==========================================

USE_CACHE = True  # Enable caching to avoid re-scraping
ENABLE_WEBSITE_ENRICHMENT = True  # Enable scraping official websites (slower)
ENABLE_PROGRESS_BAR = True  # Show progress bars during scraping
FILTER_FUTURE_ONLY = True  # Filter to only show current year and future conferences

# ==========================================
# Web Dashboard Settings
# ==========================================

WEBAPP_HOST = '0.0.0.0'  # Web dashboard host
WEBAPP_PORT = 5000  # Web dashboard port
WEBAPP_DEBUG = True  # Enable debug mode

# ==========================================
# Email Notification Settings
# ==========================================

# SMTP Configuration
EMAIL_ENABLED = os.getenv('EMAIL_ENABLED', 'false').lower() == 'true'
SMTP_HOST = os.getenv('SMTP_HOST', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USE_TLS = os.getenv('SMTP_USE_TLS', 'true').lower() == 'true'

# Email Credentials (use environment variables for security)
SENDER_EMAIL = os.getenv('SENDER_EMAIL', '')
SENDER_PASSWORD = os.getenv('SENDER_PASSWORD', '')
RECIPIENT_EMAIL = os.getenv('RECIPIENT_EMAIL', '')

# Notification Settings
DEFAULT_NOTIFICATION_DAYS = 7  # Alert for deadlines within this many days
NOTIFICATION_CATEGORIES = [
    'urgent',  # 0-3 days
    'soon',    # 4-7 days
    'upcoming' # 8-30 days
]
