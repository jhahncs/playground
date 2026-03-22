# Development Notes

## Implementation Status

✅ **Completed Components**:
- Project structure and directory setup
- Core data model (Conference dataclass)
- Logging utility
- Date parser with multiple format support
- Data validators with quality scoring
- Base scraper with rate limiting (5s), caching, retry logic
- WikiCFP scraper implementation
- WikiCFP parser (needs HTML structure refinement)
- CSV handler with deduplication
- Excel handler with formatting and hyperlinks
- Configuration system
- CLI interface
- README documentation

## Test Results

**Infrastructure Test** (2026-02-13):
- ✅ Rate limiting working (5 second intervals)
- ✅ Caching system working (saved to `data/cache/`)
- ✅ Progress bars displaying correctly
- ✅ Error handling working
- ✅ Dependencies installed successfully
- ⚠️ WikiCFP HTML parsing needs adjustment

## Known Issues

### 1. WikiCFP Parser HTML Structure

**Issue**: The current parser does not correctly extract data from WikiCFP HTML pages.

**Symptoms**:
- Text fields are concatenated without proper separation
- Dates, locations, and other fields are not being extracted correctly
- Example error: `WhenMay 8, 2026 - May 10, 2026WhereXuzhou, ChinaSubmission DeadlineFeb 28, 2026...`

**Root Cause**:
The WikiCFP HTML structure was assumed based on common patterns, but needs to be reverse-engineered from actual live pages.

**Solution**:
1. Visit http://www.wikicfp.com/cfp/ to examine the actual HTML structure
2. Update `parsers/wikicfp_parser.py` with correct selectors:
   - `parse_conference_list()`: Update table parsing logic
   - `parse_conference_detail()`: Update detail page parsing logic

**How to Debug**:
```python
# Load a cached HTML file and inspect its structure
import pickle
with open('data/cache/XXXXX.pkl', 'rb') as f:
    html = pickle.load(f)

from bs4 import BeautifulSoup
soup = BeautifulSoup(html, 'html.parser')

# Inspect the HTML structure
print(soup.prettify())

# Find table structures
tables = soup.find_all('table')
for table in tables:
    print(table.get('class'))
    print(table.get_text()[:200])
```

**Recommended Approach**:
- Use browser DevTools to inspect WikiCFP pages
- Look for consistent CSS classes or IDs
- Update the parser selectors based on actual structure
- Test with multiple conference pages to ensure robustness

## Next Steps

1. **Fix WikiCFP Parser** (Priority: High)
   - Inspect actual WikiCFP HTML structure
   - Update `parsers/wikicfp_parser.py` with correct selectors
   - Test with 10+ different conference pages

2. **Website Enrichment** (Priority: Medium)
   - Implement `scrapers/website_scraper.py` for scraping official conference sites
   - Create `parsers/cfp_extractor.py` for extracting CFP text

3. **Data Exploration Notebook** (Priority: Low)
   - Create `notebooks/exploration.ipynb`
   - Add visualization for deadlines timeline
   - Add geographic distribution analysis

4. **Testing** (Priority: High)
   - Add unit tests for parsers with mock HTML
   - Add integration tests for end-to-end workflow
   - Create test fixtures with sample WikiCFP HTML

5. **Enhancements** (Priority: Low)
   - Add scheduling support (cron job friendly)
   - Add email notifications for upcoming deadlines
   - Create web dashboard for browsing conferences

## Usage Tips

### Debugging HTML Parsing

To examine cached HTML files:
```bash
# List cached files
ls data/cache/

# Use Python to load and inspect
python3 -c "
import pickle
with open('data/cache/FILENAME.pkl', 'rb') as f:
    html = pickle.load(f)
    print(html[:1000])
"
```

### Testing Without Re-scraping

The caching system allows you to test parser changes without re-scraping:
```bash
# First run creates cache
python main.py --latest 5

# Modify parser code
# Test with cached data (no new requests)
python main.py --latest 5
```

To force re-scraping:
```bash
python main.py --clear-cache --latest 5
```

### Incremental Development

1. Start with small test (5-10 conferences)
2. Verify parsing is correct
3. Gradually increase volume
4. Use `--update` mode to add new data without duplicates

## Architecture Highlights

### Modular Design

Each component is independent:
- `scrapers/` - HTTP requests and rate limiting
- `parsers/` - HTML to data conversion
- `models/` - Data structures
- `storage/` - Persistence layer
- `utils/` - Cross-cutting concerns

### Rate Limiting

WikiCFP requires 5-second intervals between requests. This is enforced in `base_scraper.py`:
```python
self._enforce_rate_limit()  # Called before each request
```

### Caching Strategy

Responses are cached using MD5 hash of URL:
- Avoids re-scraping during development
- Speeds up repeated runs
- Stored in `data/cache/` as pickle files

### Quality Scoring

Each conference gets a score (0-1) based on completeness:
- 0.5 for required fields
- +0.1 to +0.2 for optional fields
- Filters out low-quality data

## Performance Considerations

**Current Settings**:
- Rate limit: 5 seconds per request
- Max pages per category: 5
- Max conferences per run: 200

**Estimated Times**:
- Single conference: ~5 seconds
- Category (5 pages, ~50 conferences): ~4-5 minutes
- Multiple categories: linear scaling

**Optimization Options**:
1. Adjust `MAX_PAGES_PER_CATEGORY` in config.py
2. Use `--limit` flag to cap total conferences
3. Use caching to avoid re-scraping

## Recent Updates (2026-02-13)

### ✅ Completed Features

1. **Future Conference Filtering**
   - Added `filter_future_conferences()` in validators.py
   - By default, only shows conferences from current year onwards
   - `--include-past` flag to include historical conferences

2. **Website Enrichment**
   - New `WebsiteEnricher` class in `scrapers/website_enricher.py`
   - Extracts: keynote speakers, program committee, submission tracks, registration info, social media
   - Activated with `--enrich` flag

3. **Multi-Source Support**
   - IEEE scraper: `scrapers/ieee_scraper.py`
   - ACM scraper: `scrapers/acm_scraper.py`
   - Springer scraper: `scrapers/springer_scraper.py`
   - Use `--sources "wikicfp,ieee,acm,springer"` to select sources

4. **Web Dashboard**
   - Flask-based web interface in `webapp.py`
   - Features:
     - Real-time statistics dashboard
     - Search and filtering (by source, year, quality)
     - Upcoming deadline alerts
     - CSV/Excel export
   - Run with: `python webapp.py`
   - Access at: http://localhost:5000

5. **Email Notifications**
   - New `EmailNotifier` class in `utils/email_notifier.py`
   - Features:
     - SMTP email sending with TLS/SSL support
     - Beautiful HTML email templates
     - Deadline urgency indicators (urgent, soon, upcoming)
     - Plain text fallback
     - Connection testing
   - Configuration via environment variables (.env file)
   - Supports Gmail, Outlook, Yahoo, custom SMTP
   - Activated with `--notify` flag
   - Test with: `python main.py --test-email`

### Files Added
- `scrapers/website_enricher.py` - Website enrichment functionality
- `scrapers/ieee_scraper.py` - IEEE conference scraper
- `scrapers/acm_scraper.py` - ACM conference scraper
- `scrapers/springer_scraper.py` - Springer conference scraper
- `webapp.py` - Flask web dashboard
- `templates/index.html` - Dashboard UI
- `utils/email_notifier.py` - Email notification system
- `.env.example` - Example environment variables
- `EMAIL_SETUP.md` - Email setup guide
- `QUICKSTART.md` - Quick start guide

### Files Modified
- `main.py` - Added multi-source support, enrichment, future filtering, email notifications
- `utils/validators.py` - Added `filter_future_conferences()`
- `config.py` - Added web dashboard and email settings
- `requirements.txt` - Added Flask dependency
- `README.md` - Updated documentation

## Future Enhancements Roadmap

### Phase 1: Core Improvements
- [ ] Fix WikiCFP parser HTML structure (if needed)
- [ ] Add unit tests for new scrapers
- [ ] Improve error handling for edge cases
- [ ] Validate IEEE/ACM/Springer scrapers with real data

### Phase 2: Feature Additions
- [x] Website enrichment scraper ✅
- [x] Multiple data source support (ACM, IEEE, Springer) ✅
- [ ] API endpoint for programmatic access (RESTful)
- [ ] Database backend (SQLite/PostgreSQL)

### Phase 3: User Experience
- [x] Web dashboard ✅
- [x] Email notifications for upcoming deadlines ✅
- [ ] Calendar integration (.ics export)
- [ ] User accounts and favorites
- [ ] Mobile-responsive dashboard
- [ ] Slack/Discord webhooks for notifications

### Phase 4: Advanced Features
- [ ] Machine learning for topic classification
- [ ] Recommendation system
- [ ] Conference ranking/scoring
- [ ] Historical data analysis
- [ ] Trend analysis and visualization

## Contributing

If you improve the WikiCFP parser or add features:
1. Test thoroughly with real data
2. Update this NOTES.md with findings
3. Add examples to README.md
4. Consider adding unit tests

## Contact & Support

For issues or questions:
- Check `conference_agent.log` for detailed logs
- Review cached HTML in `data/cache/` for parsing issues
- Consult README.md for usage examples
