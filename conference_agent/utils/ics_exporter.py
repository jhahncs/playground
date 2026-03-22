#!/usr/bin/env python3
"""ICS calendar export functionality."""

from datetime import datetime
from typing import List
import pandas as pd
from icalendar import Calendar, Event, vText
import pytz


class ICSExporter:
    """Export conferences to iCalendar (.ics) format."""

    def __init__(self):
        """Initialize ICS exporter."""
        self.timezone = pytz.UTC

    def create_calendar(self, conferences: List[dict], calendar_name: str = "Academic Conferences") -> Calendar:
        """
        Create an iCalendar object from a list of conferences.

        Args:
            conferences: List of conference dictionaries
            calendar_name: Name of the calendar

        Returns:
            Calendar object
        """
        cal = Calendar()
        cal.add('prodid', '-//Conference Agent//mxm.dk//')
        cal.add('version', '2.0')
        cal.add('X-WR-CALNAME', calendar_name)
        cal.add('X-WR-TIMEZONE', 'UTC')

        for conf in conferences:
            # Add conference event
            if pd.notna(conf.get('conference_start')):
                event = self._create_conference_event(conf)
                if event:
                    cal.add_component(event)

            # Add submission deadline event
            if pd.notna(conf.get('submission_deadline')):
                deadline_event = self._create_deadline_event(conf)
                if deadline_event:
                    cal.add_component(deadline_event)

        return cal

    def _create_conference_event(self, conf: dict) -> Event:
        """Create a calendar event for the conference itself."""
        try:
            event = Event()

            # Parse dates
            start_date = pd.to_datetime(conf['conference_start'])
            end_date = pd.to_datetime(conf.get('conference_end', conf['conference_start']))

            # Add one day to end date (iCal end dates are exclusive)
            end_date = end_date + pd.Timedelta(days=1)

            # Basic event info
            event.add('summary', conf.get('name', 'Conference'))
            event.add('dtstart', start_date.date())
            event.add('dtend', end_date.date())

            # Location
            location = conf.get('location', '')
            if location and str(location) != 'nan':
                event.add('location', vText(location))

            # Description
            description_parts = []

            if conf.get('acronym') and str(conf.get('acronym')) != 'nan':
                description_parts.append(f"Acronym: {conf['acronym']}")

            if conf.get('website_url') and str(conf.get('website_url')) != 'nan':
                description_parts.append(f"Website: {conf['website_url']}")

            if conf.get('cfp_url') and str(conf.get('cfp_url')) != 'nan':
                description_parts.append(f"CFP: {conf['cfp_url']}")

            if conf.get('topics') and str(conf.get('topics')) != 'nan':
                description_parts.append(f"Topics: {conf['topics']}")

            if description_parts:
                event.add('description', '\n'.join(description_parts))

            # URL
            if conf.get('website_url') and str(conf.get('website_url')) != 'nan':
                event.add('url', vText(conf['website_url']))

            # Unique ID
            uid = f"{conf.get('conference_id', conf.get('name', 'unknown'))}@conferenceagent"
            event.add('uid', uid)

            # Timestamp
            event.add('dtstamp', datetime.now(self.timezone))

            return event

        except Exception as e:
            print(f"Error creating conference event for {conf.get('name')}: {e}")
            return None

    def _create_deadline_event(self, conf: dict) -> Event:
        """Create a calendar event for the submission deadline."""
        try:
            deadline = pd.to_datetime(conf['submission_deadline'])

            event = Event()
            event.add('summary', f"📝 Deadline: {conf.get('acronym', conf.get('name', 'Conference'))}")
            event.add('dtstart', deadline.date())
            event.add('dtend', (deadline + pd.Timedelta(days=1)).date())

            # Description
            description_parts = [
                f"Submission deadline for: {conf.get('name', 'Conference')}",
            ]

            if conf.get('website_url') and str(conf.get('website_url')) != 'nan':
                description_parts.append(f"Website: {conf['website_url']}")

            if conf.get('cfp_url') and str(conf.get('cfp_url')) != 'nan':
                description_parts.append(f"CFP: {conf['cfp_url']}")

            event.add('description', '\n'.join(description_parts))

            # Add alarm (reminder 7 days before)
            from icalendar import Alarm
            alarm = Alarm()
            alarm.add('action', 'DISPLAY')
            alarm.add('description', f"Submission deadline approaching for {conf.get('acronym', conf.get('name'))}")
            alarm.add('trigger', '-P7D')  # 7 days before
            event.add_component(alarm)

            # Unique ID
            uid = f"{conf.get('conference_id', conf.get('name', 'unknown'))}-deadline@conferenceagent"
            event.add('uid', uid)

            # Timestamp
            event.add('dtstamp', datetime.now(self.timezone))

            return event

        except Exception as e:
            print(f"Error creating deadline event for {conf.get('name')}: {e}")
            return None

    def export_to_file(self, conferences: List[dict], output_path: str, calendar_name: str = "Academic Conferences"):
        """
        Export conferences to an .ics file.

        Args:
            conferences: List of conference dictionaries
            output_path: Path to save the .ics file
            calendar_name: Name of the calendar
        """
        cal = self.create_calendar(conferences, calendar_name)

        with open(output_path, 'wb') as f:
            f.write(cal.to_ical())

    def export_to_string(self, conferences: List[dict], calendar_name: str = "Academic Conferences") -> str:
        """
        Export conferences to an .ics string.

        Args:
            conferences: List of conference dictionaries
            calendar_name: Name of the calendar

        Returns:
            iCalendar string
        """
        cal = self.create_calendar(conferences, calendar_name)
        return cal.to_ical().decode('utf-8')
