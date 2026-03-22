# Email Notification Setup Guide

Conference Agent can send email notifications for upcoming conference deadlines. This guide will help you set up email notifications.

## Quick Setup

### 1. Copy Environment File

```bash
cp .env.example .env
```

### 2. Configure Email Settings

Edit `.env` and set your email credentials:

```bash
EMAIL_ENABLED=true
SENDER_EMAIL=your-email@gmail.com
SENDER_PASSWORD=your-app-password
RECIPIENT_EMAIL=where-to-send@example.com
```

### 3. Test Email Configuration

```bash
python main.py --test-email
```

You should see: ✅ Email configuration is valid!

### 4. Send Notifications

```bash
# Scrape conferences and send email for deadlines within 7 days
python main.py --latest 50 --notify

# Custom threshold (e.g., 14 days)
python main.py --latest 50 --notify --notify-days 14
```

## Gmail Setup (Recommended)

### Step 1: Enable 2-Step Verification

1. Go to [Google Account Security](https://myaccount.google.com/security)
2. Enable "2-Step Verification"

### Step 2: Generate App Password

1. Go to [App Passwords](https://myaccount.google.com/apppasswords)
2. Select app: "Mail"
3. Select device: "Other (Custom name)" → Enter "Conference Agent"
4. Click "Generate"
5. Copy the 16-character password

### Step 3: Configure .env

```bash
EMAIL_ENABLED=true
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USE_TLS=true
SENDER_EMAIL=your-gmail@gmail.com
SENDER_PASSWORD=xxxx xxxx xxxx xxxx  # 16-character app password
RECIPIENT_EMAIL=your-gmail@gmail.com  # Can be the same or different
```

## Other Email Providers

### Outlook / Office 365

```bash
SMTP_HOST=smtp-mail.outlook.com
SMTP_PORT=587
SMTP_USE_TLS=true
SENDER_EMAIL=your-email@outlook.com
SENDER_PASSWORD=your-password
```

### Yahoo Mail

```bash
SMTP_HOST=smtp.mail.yahoo.com
SMTP_PORT=587
SMTP_USE_TLS=true
SENDER_EMAIL=your-email@yahoo.com
SENDER_PASSWORD=your-app-password  # Generate at Yahoo Account Security
```

### Custom SMTP Server

```bash
SMTP_HOST=mail.yourdomain.com
SMTP_PORT=587
SMTP_USE_TLS=true
SENDER_EMAIL=noreply@yourdomain.com
SENDER_PASSWORD=your-password
```

## Email Notification Features

### What's Included in Notifications

- **Deadline Countdown**: Days remaining until submission deadline
- **Conference Details**: Name, location, dates
- **Important Dates**: Submission, notification, camera-ready deadlines
- **Links**: Official website and CFP URLs
- **Visual Indicators**: Color-coded urgency (urgent: 0-3 days, soon: 4-7 days)

### Notification Thresholds

- **Urgent** (Red): Deadlines within 3 days
- **Soon** (Orange): Deadlines within 4-7 days
- **Upcoming** (Green): Deadlines within 8-30 days

## Usage Examples

### Daily Notification (Cron Job)

Add to crontab:

```bash
# Every day at 9 AM, check for deadlines in next 7 days
0 9 * * * cd /home/jhahn/playground/conference_agent && \
  /usr/bin/python3 main.py --latest 100 --notify --update >> cron.log 2>&1
```

### Weekly Digest

```bash
# Every Monday at 8 AM, send 30-day outlook
0 8 * * 1 cd /home/jhahn/playground/conference_agent && \
  /usr/bin/python3 main.py --latest 200 --notify --notify-days 30 --update
```

### Urgent Alerts Only

```bash
# Check multiple times per day for very urgent deadlines
0 */4 * * * cd /home/jhahn/playground/conference_agent && \
  /usr/bin/python3 main.py --latest 50 --notify --notify-days 3 --update
```

## Troubleshooting

### "SMTP authentication failed"

**Problem**: Invalid email or password

**Solutions**:
- For Gmail: Make sure you're using an App Password, not your regular password
- Check that 2-Step Verification is enabled
- Verify email and password are correct in .env

### "Email is disabled"

**Problem**: EMAIL_ENABLED is not set to true

**Solution**:
```bash
EMAIL_ENABLED=true
```

### "Connection timeout"

**Problem**: Firewall or network blocking SMTP

**Solutions**:
- Check firewall settings
- Try different port (465 for SSL, 587 for TLS)
- Verify SMTP_HOST is correct

### "No conferences with deadlines"

**Problem**: No upcoming deadlines found

**Solution**:
- Increase notification threshold: `--notify-days 30`
- Run scraper first to collect data

### Testing Email

```bash
# Test connection without scraping
python main.py --test-email

# If successful, try a real notification
python main.py --latest 10 --notify
```

## Security Best Practices

### 1. Never Commit .env

The `.env` file contains sensitive credentials. Make sure it's in `.gitignore`:

```bash
echo ".env" >> .gitignore
```

### 2. Use App Passwords

- Never use your main email password
- Use app-specific passwords that can be revoked
- Generate separate passwords for different apps

### 3. Limit Permissions

- Use a dedicated email account for sending (e.g., noreply@yourdomain.com)
- Don't use personal accounts for automated systems

### 4. Environment Variables

For production, use environment variables instead of .env:

```bash
export EMAIL_ENABLED=true
export SENDER_EMAIL=noreply@example.com
export SENDER_PASSWORD=secure-password
export RECIPIENT_EMAIL=admin@example.com

python main.py --latest 100 --notify
```

## Email Notification API

You can also use the EmailNotifier programmatically:

```python
from utils.email_notifier import EmailNotifier
from models.conference import Conference

# Initialize notifier
notifier = EmailNotifier(
    smtp_host='smtp.gmail.com',
    smtp_port=587,
    sender_email='sender@gmail.com',
    sender_password='app-password',
    use_tls=True
)

# Test connection
if notifier.test_connection():
    print("Connection successful!")

# Send notification
conferences = [...]  # List of Conference objects
notifier.send_deadline_notification(
    recipient_email='recipient@example.com',
    conferences=conferences,
    days_threshold=7
)
```

## Support

For issues with email notifications:
1. Check `conference_agent.log` for error messages
2. Run `--test-email` to diagnose connection issues
3. Verify SMTP settings with your email provider
4. Check firewall and network settings
