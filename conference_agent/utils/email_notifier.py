"""Email notification utility for conference deadlines."""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List
from datetime import datetime, timedelta
from pathlib import Path

from models.conference import Conference
from utils.logger import get_logger

logger = get_logger(__name__)


class EmailNotifier:
    """Send email notifications for upcoming conference deadlines."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        sender_email: str,
        sender_password: str,
        use_tls: bool = True
    ):
        """
        Initialize email notifier.

        Args:
            smtp_host: SMTP server host (e.g., smtp.gmail.com)
            smtp_port: SMTP server port (e.g., 587 for TLS)
            sender_email: Sender email address
            sender_password: Sender email password or app password
            use_tls: Whether to use TLS encryption
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.use_tls = use_tls

    def send_deadline_notification(
        self,
        recipient_email: str,
        conferences: List[Conference],
        days_threshold: int = 7
    ) -> bool:
        """
        Send email notification for conferences with upcoming deadlines.

        Args:
            recipient_email: Recipient email address
            conferences: List of conferences to check
            days_threshold: Alert for deadlines within this many days

        Returns:
            True if email sent successfully, False otherwise
        """
        # Filter conferences with upcoming deadlines
        upcoming = self._filter_upcoming_deadlines(conferences, days_threshold)

        if not upcoming:
            logger.info(f"No conferences with deadlines within {days_threshold} days")
            return False

        # Generate email content
        subject = f"🎓 {len(upcoming)} Conference Deadline(s) in Next {days_threshold} Days"
        html_content = self._generate_html_email(upcoming, days_threshold)
        text_content = self._generate_text_email(upcoming, days_threshold)

        # Send email
        return self._send_email(recipient_email, subject, html_content, text_content)

    def _filter_upcoming_deadlines(
        self,
        conferences: List[Conference],
        days_threshold: int
    ) -> List[Conference]:
        """Filter conferences with deadlines within threshold."""
        today = datetime.now().date()
        threshold_date = today + timedelta(days=days_threshold)

        upcoming = []
        for conf in conferences:
            if conf.submission_deadline:
                if today <= conf.submission_deadline <= threshold_date:
                    upcoming.append(conf)

        # Sort by deadline
        upcoming.sort(key=lambda c: c.submission_deadline)
        return upcoming

    def _generate_html_email(
        self,
        conferences: List[Conference],
        days_threshold: int
    ) -> str:
        """Generate HTML email content."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }}
        .conference-card {{
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
        }}
        .conference-title {{
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }}
        .urgent {{
            border-left-color: #ff4757;
        }}
        .soon {{
            border-left-color: #ffa502;
        }}
        .deadline {{
            color: #ff4757;
            font-weight: bold;
            font-size: 1.1em;
        }}
        .info-row {{
            margin: 8px 0;
        }}
        .label {{
            font-weight: bold;
            color: #666;
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
        a {{
            color: #667eea;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎓 Conference Deadline Reminder</h1>
        <p>{len(conferences)} conference(s) with deadlines in the next {days_threshold} days</p>
    </div>
"""

        for conf in conferences:
            days_left = (conf.submission_deadline - datetime.now().date()).days

            # Determine urgency class
            urgency_class = "urgent" if days_left <= 3 else "soon" if days_left <= 7 else ""

            html += f"""
    <div class="conference-card {urgency_class}">
        <div class="conference-title">
            {conf.name}
            {f'({conf.acronym})' if conf.acronym else ''}
        </div>
        <div class="deadline">
            ⏰ Deadline: {conf.submission_deadline.strftime('%B %d, %Y')} ({days_left} days left)
        </div>
        <div class="info-row">
            <span class="label">📍 Location:</span> {conf.location}
        </div>
        <div class="info-row">
            <span class="label">📅 Conference Dates:</span>
            {conf.conference_start.strftime('%B %d, %Y') if conf.conference_start else 'TBD'}
            {f' - {conf.conference_end.strftime("%B %d, %Y")}' if conf.conference_end else ''}
        </div>
"""
            if conf.notification_date:
                html += f"""
        <div class="info-row">
            <span class="label">📢 Notification:</span> {conf.notification_date.strftime('%B %d, %Y')}
        </div>
"""
            if conf.website_url:
                html += f"""
        <div class="info-row">
            <span class="label">🔗 Website:</span> <a href="{conf.website_url}" target="_blank">{conf.website_url}</a>
        </div>
"""
            if conf.cfp_url:
                html += f"""
        <div class="info-row">
            <span class="label">📄 CFP:</span> <a href="{conf.cfp_url}" target="_blank">View Call for Papers</a>
        </div>
"""
            html += "    </div>\n"

        html += """
    <div class="footer">
        <p>This notification was generated by Conference Agent</p>
        <p>📊 <a href="http://localhost:5000">Open Dashboard</a></p>
    </div>
</body>
</html>
"""
        return html

    def _generate_text_email(
        self,
        conferences: List[Conference],
        days_threshold: int
    ) -> str:
        """Generate plain text email content."""
        text = f"Conference Deadline Reminder\n"
        text += f"=" * 50 + "\n\n"
        text += f"{len(conferences)} conference(s) with deadlines in the next {days_threshold} days\n\n"

        for conf in conferences:
            days_left = (conf.submission_deadline - datetime.now().date()).days

            text += f"\n{conf.name}"
            if conf.acronym:
                text += f" ({conf.acronym})"
            text += "\n" + "-" * 50 + "\n"

            text += f"⏰ Deadline: {conf.submission_deadline.strftime('%B %d, %Y')} ({days_left} days left)\n"
            text += f"📍 Location: {conf.location}\n"

            if conf.conference_start:
                text += f"📅 Conference: {conf.conference_start.strftime('%B %d, %Y')}"
                if conf.conference_end:
                    text += f" - {conf.conference_end.strftime('%B %d, %Y')}"
                text += "\n"

            if conf.notification_date:
                text += f"📢 Notification: {conf.notification_date.strftime('%B %d, %Y')}\n"

            if conf.website_url:
                text += f"🔗 Website: {conf.website_url}\n"

            if conf.cfp_url:
                text += f"📄 CFP: {conf.cfp_url}\n"

            text += "\n"

        text += "\n" + "=" * 50 + "\n"
        text += "Generated by Conference Agent\n"

        return text

    def _send_email(
        self,
        recipient_email: str,
        subject: str,
        html_content: str,
        text_content: str
    ) -> bool:
        """
        Send email via SMTP.

        Args:
            recipient_email: Recipient email address
            subject: Email subject
            html_content: HTML email body
            text_content: Plain text email body

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Create message
            message = MIMEMultipart('alternative')
            message['Subject'] = subject
            message['From'] = self.sender_email
            message['To'] = recipient_email

            # Attach text and HTML parts
            text_part = MIMEText(text_content, 'plain')
            html_part = MIMEText(html_content, 'html')

            message.attach(text_part)
            message.attach(html_part)

            # Send via SMTP
            logger.info(f"Connecting to SMTP server {self.smtp_host}:{self.smtp_port}")

            if self.use_tls:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port)
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port)

            server.login(self.sender_email, self.sender_password)
            server.send_message(message)
            server.quit()

            logger.info(f"Email sent successfully to {recipient_email}")
            return True

        except smtplib.SMTPAuthenticationError:
            logger.error("SMTP authentication failed. Check email and password.")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def test_connection(self) -> bool:
        """
        Test SMTP connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"Testing SMTP connection to {self.smtp_host}:{self.smtp_port}")

            if self.use_tls:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10)
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, timeout=10)

            server.login(self.sender_email, self.sender_password)
            server.quit()

            logger.info("SMTP connection test successful")
            return True

        except Exception as e:
            logger.error(f"SMTP connection test failed: {e}")
            return False
