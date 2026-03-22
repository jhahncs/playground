#!/usr/bin/env python3
"""Main entry point for conference agent."""

import argparse
import sys
from typing import List

from scrapers.wikicfp_scraper import WikiCFPScraper
from scrapers.ieee_scraper import IEEEScraper
from scrapers.acm_scraper import ACMScraper
from scrapers.springer_scraper import SpringerScraper
from scrapers.website_enricher import WebsiteEnricher
from storage.csv_handler import CSVHandler
from storage.excel_handler import ExcelHandler
from utils.logger import setup_logger, get_logger
from utils.validators import filter_valid_conferences, calculate_quality_score, filter_future_conferences, filter_conferences_by_date_range, filter_conferences_by_multiple_date_ranges, deduplicate_conferences
from utils.email_notifier import EmailNotifier
import config

# Setup logger
logger = setup_logger('conference_agent', log_file=config.LOG_FILE)


class ConferenceAgent:
    """Main conference agent orchestrator."""

    def __init__(self):
        """Initialize the conference agent."""
        self.scrapers = {
            'wikicfp': WikiCFPScraper(
                rate_limit=config.RATE_LIMIT_SECONDS,
                max_retries=config.MAX_RETRIES,
                cache_dir=config.CACHE_DIR,
                use_cache=config.USE_CACHE
            ),
            'ieee': IEEEScraper(
                rate_limit=config.RATE_LIMIT_SECONDS,
                max_retries=config.MAX_RETRIES,
                cache_dir=config.CACHE_DIR,
                use_cache=config.USE_CACHE
            ),
            'acm': ACMScraper(
                rate_limit=config.RATE_LIMIT_SECONDS,
                max_retries=config.MAX_RETRIES,
                cache_dir=config.CACHE_DIR,
                use_cache=config.USE_CACHE
            ),
            'springer': SpringerScraper(
                rate_limit=config.RATE_LIMIT_SECONDS,
                max_retries=config.MAX_RETRIES,
                cache_dir=config.CACHE_DIR,
                use_cache=config.USE_CACHE
            )
        }
        self.enricher = WebsiteEnricher()
        self.csv_handler = CSVHandler(config.CSV_OUTPUT_PATH)
        self.excel_handler = ExcelHandler(config.EXCEL_OUTPUT_PATH)

    def run(self, args):
        """
        Run the conference agent.

        Args:
            args: Command line arguments
        """
        logger.info("=== Conference Agent Started ===")

        conferences = []

        # Determine which sources to use
        sources = args.sources.split(',') if args.sources else ['wikicfp']
        logger.info(f"Using sources: {sources}")

        # Scrape by categories
        if args.categories:
            categories = [c.strip() for c in args.categories.split(',')]
            logger.info(f"Scraping categories: {categories}")

            for source in sources:
                if source == 'wikicfp':
                    for category in categories:
                        logger.info(f"Scraping category from WikiCFP: {category}")
                        cat_conferences = self.scrapers['wikicfp'].scrape_category(
                            category,
                            max_pages=args.max_pages or config.MAX_PAGES_PER_CATEGORY
                        )
                        conferences.extend(cat_conferences)

        # Scrape latest CFPs
        elif args.latest:
            logger.info(f"Scraping latest {args.latest} CFPs")

            for source in sources:
                if source == 'wikicfp':
                    conferences.extend(self.scrapers['wikicfp'].scrape_latest_cfps(limit=args.latest))
                elif source == 'ieee':
                    conferences.extend(self.scrapers['ieee'].scrape_upcoming_conferences(limit=args.latest))
                elif source == 'acm':
                    conferences.extend(self.scrapers['acm'].scrape_upcoming_conferences(limit=args.latest))
                elif source == 'springer':
                    conferences.extend(self.scrapers['springer'].scrape_upcoming_conferences(limit=args.latest))

        # Search by keyword(s) - OR logic (comma-separated)
        elif args.search:
            keywords = [kw.strip() for kw in args.search.split(',') if kw.strip()]
            logger.info(f"Searching for keywords (OR): {keywords}")
            per_keyword_limit = args.limit or 50
            conferences = []
            for kw in keywords:
                results = self.scrapers['wikicfp'].search_by_keyword(kw, limit=per_keyword_limit)
                conferences.extend(results)
            total_before = len(conferences)
            conferences = deduplicate_conferences(conferences)
            logger.info(f"Combined OR search: {len(conferences)} unique conferences from {len(keywords)} keyword(s) (removed {total_before - len(conferences)} duplicates)")

        else:
            logger.error("No action specified. Use --categories, --latest, or --search")
            return

        logger.info(f"Scraped {len(conferences)} conferences total")

        # Filter by multiple date ranges if specified
        if args.date_ranges:
            from datetime import datetime
            try:
                date_ranges = []
                for range_str in args.date_ranges.split(','):
                    range_str = range_str.strip()
                    if ':' not in range_str:
                        logger.error(f"Invalid date range format: '{range_str}'. Expected format: 'YYYY-MM-DD:YYYY-MM-DD'")
                        return

                    start_str, end_str = range_str.split(':', 1)
                    start_dt = datetime.strptime(start_str.strip(), '%Y-%m-%d').date()
                    end_dt = datetime.strptime(end_str.strip(), '%Y-%m-%d').date()

                    if start_dt > end_dt:
                        logger.error(f"Invalid date range: start date {start_str} is after end date {end_str}")
                        return

                    date_ranges.append((start_dt, end_dt))

                conferences, filtered = filter_conferences_by_multiple_date_ranges(conferences, date_ranges)
                logger.info(f"Filtered to conferences within {len(date_ranges)} date range(s): {len(conferences)} conferences")
                for start_dt, end_dt in date_ranges:
                    logger.info(f"  Range: {start_dt} to {end_dt}")

            except ValueError as e:
                logger.error(f"Error parsing date ranges: {e}")
                return

        # Filter by single date range if specified
        elif args.start_date or args.end_date:
            from datetime import datetime
            start_dt = datetime.strptime(args.start_date, '%Y-%m-%d').date() if args.start_date else None
            end_dt = datetime.strptime(args.end_date, '%Y-%m-%d').date() if args.end_date else None

            conferences, filtered = filter_conferences_by_date_range(conferences, start_dt, end_dt)
            if start_dt and end_dt:
                logger.info(f"Filtered to conferences between {args.start_date} and {args.end_date}: {len(conferences)} conferences")
            elif start_dt:
                logger.info(f"Filtered to conferences from {args.start_date} onwards: {len(conferences)} conferences")
            elif end_dt:
                logger.info(f"Filtered to conferences until {args.end_date}: {len(conferences)} conferences")
        # Filter to only include current year and future conferences
        elif not args.include_past:
            conferences, past = filter_future_conferences(conferences)
            logger.info(f"Keeping only current and future conferences: {len(conferences)} conferences")

        # Calculate quality scores
        for conf in conferences:
            conf.data_quality_score = calculate_quality_score(conf)

        # Enrich from official websites if enabled
        if args.enrich and config.ENABLE_WEBSITE_ENRICHMENT:
            logger.info("Enriching conferences from official websites")
            for conf in conferences:
                if conf.website_url and conf.data_quality_score < config.ENRICH_THRESHOLD:
                    logger.info(f"Enriching: {conf.name}")
                    enriched_data = self.enricher.enrich_from_website(conf.website_url)
                    # You can extend the Conference model to include enriched data
                    # For now, we'll log what we found
                    if enriched_data.get('keynote_speakers'):
                        logger.info(f"  Found {len(enriched_data['keynote_speakers'])} keynote speakers")

        # Filter by quality
        valid, invalid = filter_valid_conferences(
            conferences,
            min_quality=args.min_quality or config.MIN_QUALITY_SCORE
        )

        logger.info(f"Valid conferences: {len(valid)}")
        logger.info(f"Invalid/low-quality conferences: {len(invalid)}")

        if not valid:
            logger.warning("No valid conferences found")
            return

        # Save to CSV
        if args.update:
            logger.info("Updating existing data with deduplication")
            self.csv_handler.update_conferences(valid)
        else:
            logger.info("Saving to CSV (overwrite mode)")
            self.csv_handler.save_conferences(valid, mode='overwrite')

        # Export to Excel if requested
        if args.excel or args.output.endswith('.xlsx'):
            logger.info("Exporting to Excel")
            excel_path = args.output if args.output.endswith('.xlsx') else config.EXCEL_OUTPUT_PATH
            excel_handler = ExcelHandler(excel_path)
            excel_handler.export_conferences(valid)

        # Export summary
        self.csv_handler.export_summary(config.SUMMARY_OUTPUT_PATH)

        # Print statistics
        stats = self.csv_handler.get_statistics()
        logger.info("=== Statistics ===")
        logger.info(f"Total conferences: {stats['total']}")
        logger.info(f"Average quality score: {stats['avg_quality']:.2f}")

        # Send email notification if enabled
        if args.notify:
            self._send_email_notification(valid, args.notify_days or config.DEFAULT_NOTIFICATION_DAYS)

        logger.info("=== Conference Agent Completed ===")

    def _send_email_notification(self, conferences: List, days_threshold: int):
        """Send email notification for upcoming deadlines."""
        if not config.EMAIL_ENABLED:
            logger.warning("Email notifications are disabled. Set EMAIL_ENABLED=true in environment.")
            return

        if not config.SENDER_EMAIL or not config.SENDER_PASSWORD:
            logger.error("Email credentials not configured. Set SENDER_EMAIL and SENDER_PASSWORD.")
            return

        if not config.RECIPIENT_EMAIL:
            logger.error("Recipient email not configured. Set RECIPIENT_EMAIL.")
            return

        try:
            notifier = EmailNotifier(
                smtp_host=config.SMTP_HOST,
                smtp_port=config.SMTP_PORT,
                sender_email=config.SENDER_EMAIL,
                sender_password=config.SENDER_PASSWORD,
                use_tls=config.SMTP_USE_TLS
            )

            logger.info(f"Sending email notification to {config.RECIPIENT_EMAIL}")
            success = notifier.send_deadline_notification(
                recipient_email=config.RECIPIENT_EMAIL,
                conferences=conferences,
                days_threshold=days_threshold
            )

            if success:
                logger.info("Email notification sent successfully")
            else:
                logger.warning("Failed to send email notification")

        except Exception as e:
            logger.error(f"Error sending email notification: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Conference information collection agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape machine learning and AI conferences
  python main.py --categories "machine learning,artificial intelligence" --output conferences.csv

  # Scrape latest 50 CFPs
  python main.py --latest 50 --excel

  # Search for specific keyword(s) (OR-combined)
  python main.py --search "deep learning,NLP" --limit 30

  # Update existing data
  python main.py --categories "computer vision" --update

  # Collect conferences in a specific date range (e.g., March-June 2026)
  python main.py --latest 100 --start-date 2026-03-01 --end-date 2026-06-30

  # Collect conferences from a specific date onwards
  python main.py --search "NLP,machine learning" --start-date 2026-05-01

  # Collect conferences in multiple date ranges (e.g., Mar-Jun OR Sep-Nov 2026)
  python main.py --latest 200 --date-ranges "2026-03-01:2026-06-30,2026-09-01:2026-11-30"
        """
    )

    # Scraping options
    scrape_group = parser.add_mutually_exclusive_group()
    scrape_group.add_argument(
        '--categories',
        type=str,
        help='Comma-separated list of categories to scrape (e.g., "machine learning,computer vision")'
    )
    scrape_group.add_argument(
        '--latest',
        type=int,
        help='Scrape N latest CFPs'
    )
    scrape_group.add_argument(
        '--search',
        type=str,
        help='Search by keyword(s), comma-separated for OR search (e.g., "deep learning,NLP,CV")'
    )

    # Scraping parameters
    parser.add_argument(
        '--max-pages',
        type=int,
        help=f'Maximum pages per category (default: {config.MAX_PAGES_PER_CATEGORY})'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Maximum number of conferences to scrape'
    )
    parser.add_argument(
        '--sources',
        type=str,
        default='wikicfp',
        help='Comma-separated list of sources (wikicfp,ieee,acm,springer). Default: wikicfp'
    )

    # Quality filtering
    parser.add_argument(
        '--min-quality',
        type=float,
        help=f'Minimum quality score (0-1, default: {config.MIN_QUALITY_SCORE})'
    )
    parser.add_argument(
        '--include-past',
        action='store_true',
        help='Include past conferences (by default only current year and future conferences are included)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Filter conferences starting from this date (format: YYYY-MM-DD). Overrides --include-past.'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='Filter conferences until this date (format: YYYY-MM-DD). Overrides --include-past.'
    )
    parser.add_argument(
        '--date-ranges',
        type=str,
        help='Filter conferences by multiple date ranges (format: "YYYY-MM-DD:YYYY-MM-DD,YYYY-MM-DD:YYYY-MM-DD"). '
             'Example: "2026-03-01:2026-06-30,2026-09-01:2026-11-30" for conferences in Mar-Jun or Sep-Nov. '
             'Overrides --start-date, --end-date, and --include-past.'
    )
    parser.add_argument(
        '--enrich',
        action='store_true',
        help='Enrich data by scraping official conference websites (slower)'
    )

    # Email notification options
    parser.add_argument(
        '--notify',
        action='store_true',
        help='Send email notification for upcoming deadlines'
    )
    parser.add_argument(
        '--notify-days',
        type=int,
        help=f'Number of days threshold for notifications (default: {config.DEFAULT_NOTIFICATION_DAYS})'
    )
    parser.add_argument(
        '--test-email',
        action='store_true',
        help='Test email configuration and send a test email'
    )

    # Output options
    parser.add_argument(
        '--output',
        type=str,
        default=config.CSV_OUTPUT_PATH,
        help='Output file path (CSV or Excel)'
    )
    parser.add_argument(
        '--excel',
        action='store_true',
        help='Also export to Excel with formatting'
    )
    parser.add_argument(
        '--update',
        action='store_true',
        help='Update existing data (merge and deduplicate)'
    )

    # Cache options
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear cache before running'
    )

    args = parser.parse_args()

    # Test email configuration if requested
    if args.test_email:
        logger.info("Testing email configuration...")
        if not config.EMAIL_ENABLED:
            logger.error("Email is disabled. Set EMAIL_ENABLED=true in environment.")
            sys.exit(1)

        if not config.SENDER_EMAIL or not config.SENDER_PASSWORD:
            logger.error("Email credentials not configured. Set SENDER_EMAIL and SENDER_PASSWORD.")
            sys.exit(1)

        try:
            notifier = EmailNotifier(
                smtp_host=config.SMTP_HOST,
                smtp_port=config.SMTP_PORT,
                sender_email=config.SENDER_EMAIL,
                sender_password=config.SENDER_PASSWORD,
                use_tls=config.SMTP_USE_TLS
            )

            if notifier.test_connection():
                logger.info("✅ Email configuration is valid!")
                logger.info(f"SMTP: {config.SMTP_HOST}:{config.SMTP_PORT}")
                logger.info(f"Sender: {config.SENDER_EMAIL}")
            else:
                logger.error("❌ Email configuration test failed")
                sys.exit(1)

        except Exception as e:
            logger.error(f"Email test error: {e}")
            sys.exit(1)

        sys.exit(0)

    # Create and run agent
    agent = ConferenceAgent()

    if args.clear_cache:
        logger.info("Clearing cache for all scrapers")
        for scraper in agent.scrapers.values():
            scraper.clear_cache()

    try:
        agent.run(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
