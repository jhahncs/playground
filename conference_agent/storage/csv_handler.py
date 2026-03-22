"""CSV storage handler for conference data."""

import pandas as pd
from typing import List, Optional
from models.conference import Conference
from utils.logger import get_logger
from utils.validators import deduplicate_conferences
import os

logger = get_logger(__name__)


class CSVHandler:
    """Handle reading and writing conference data to CSV."""

    def __init__(self, output_path: str = "data/processed/conferences.csv"):
        """
        Initialize CSV handler.

        Args:
            output_path: Path to CSV file
        """
        self.output_path = output_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def save_conferences(self, conferences: List[Conference], mode: str = 'overwrite') -> bool:
        """
        Save conferences to CSV file.

        Args:
            conferences: List of Conference objects
            mode: 'overwrite' or 'append'

        Returns:
            True if successful
        """
        if not conferences:
            logger.warning("No conferences to save")
            return False

        try:
            # Convert to dictionaries
            data = [conf.to_dict() for conf in conferences]

            # Create DataFrame
            df = pd.DataFrame(data)

            if mode == 'append' and os.path.exists(self.output_path):
                # Load existing data and append
                logger.info(f"Appending to existing file: {self.output_path}")
                existing_df = pd.read_csv(self.output_path, encoding='utf-8')
                df = pd.concat([existing_df, df], ignore_index=True)

            # Save to CSV with UTF-8 encoding
            df.to_csv(self.output_path, index=False, encoding='utf-8')

            logger.info(f"Saved {len(conferences)} conferences to {self.output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save conferences to CSV: {e}")
            return False

    def load_conferences(self) -> List[Conference]:
        """
        Load conferences from CSV file.

        Returns:
            List of Conference objects
        """
        if not os.path.exists(self.output_path):
            logger.warning(f"CSV file not found: {self.output_path}")
            return []

        try:
            df = pd.read_csv(self.output_path, encoding='utf-8')

            conferences = []

            for _, row in df.iterrows():
                try:
                    # Convert row to dictionary
                    data = row.to_dict()

                    # Handle NaN values
                    data = {k: (v if pd.notna(v) else None) for k, v in data.items()}

                    # Create Conference object
                    conference = Conference.from_dict(data)
                    conferences.append(conference)

                except Exception as e:
                    logger.warning(f"Failed to parse conference row: {e}")
                    continue

            logger.info(f"Loaded {len(conferences)} conferences from {self.output_path}")
            return conferences

        except Exception as e:
            logger.error(f"Failed to load conferences from CSV: {e}")
            return []

    def update_conferences(self, new_conferences: List[Conference]) -> bool:
        """
        Update existing CSV with new conferences (merge and deduplicate).

        Args:
            new_conferences: List of new Conference objects

        Returns:
            True if successful
        """
        logger.info("Updating conferences with deduplication")

        # Load existing conferences
        existing = self.load_conferences()

        # Combine and deduplicate
        all_conferences = existing + new_conferences
        deduplicated = deduplicate_conferences(all_conferences)

        logger.info(f"After deduplication: {len(deduplicated)} unique conferences")

        # Save
        return self.save_conferences(deduplicated, mode='overwrite')

    def get_statistics(self) -> dict:
        """
        Get statistics about stored conferences.

        Returns:
            Dictionary with statistics
        """
        conferences = self.load_conferences()

        if not conferences:
            return {
                'total': 0,
                'avg_quality': 0.0,
                'by_year': {},
                'by_location': {}
            }

        # Calculate statistics
        quality_scores = [c.data_quality_score for c in conferences]
        years = [c.year for c in conferences if c.year]
        locations = [c.location for c in conferences if c.location]

        # Count by year
        year_counts = {}
        for year in years:
            year_counts[year] = year_counts.get(year, 0) + 1

        # Count by location
        location_counts = {}
        for loc in locations:
            location_counts[loc] = location_counts.get(loc, 0) + 1

        stats = {
            'total': len(conferences),
            'avg_quality': sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
            'by_year': dict(sorted(year_counts.items())),
            'by_location': dict(sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        }

        return stats

    def export_summary(self, output_path: str = "data/processed/summary.txt"):
        """
        Export summary statistics to text file.

        Args:
            output_path: Path to summary file
        """
        stats = self.get_statistics()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=== Conference Data Summary ===\n\n")
            f.write(f"Total conferences: {stats['total']}\n")
            f.write(f"Average quality score: {stats['avg_quality']:.2f}\n\n")

            f.write("Conferences by year:\n")
            for year, count in stats['by_year'].items():
                f.write(f"  {year}: {count}\n")

            f.write("\nTop 10 locations:\n")
            for loc, count in stats['by_location'].items():
                f.write(f"  {loc}: {count}\n")

        logger.info(f"Exported summary to {output_path}")
