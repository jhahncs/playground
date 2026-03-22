#!/usr/bin/env python3
"""Web dashboard for conference agent."""

from flask import Flask, render_template, jsonify, request, send_file, Response
import pandas as pd
import os
from datetime import datetime, timedelta
import tempfile

import config
from storage.csv_handler import CSVHandler
from utils.logger import setup_logger, get_logger
from utils.ics_exporter import ICSExporter

logger = setup_logger('conference_webapp', log_file='webapp.log')

app = Flask(__name__)
csv_handler = CSVHandler(config.CSV_OUTPUT_PATH)
ics_exporter = ICSExporter()


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')


@app.route('/api/conferences')
def get_conferences():
    """Get all conferences as JSON."""
    try:
        if not os.path.exists(config.CSV_OUTPUT_PATH):
            return jsonify({'error': 'No data available. Please run the scraper first.'}), 404

        df = pd.read_csv(config.CSV_OUTPUT_PATH)

        # Parse conference dates before filtering
        df['conference_start'] = pd.to_datetime(df['conference_start'], errors='coerce')
        df['conference_end'] = pd.to_datetime(df['conference_end'], errors='coerce')

        # Filter by query parameters
        source = request.args.get('source')
        year = request.args.get('year', type=int)
        min_quality = request.args.get('min_quality', type=float)
        start_date = request.args.get('start_date')  # Format: YYYY-MM-DD
        end_date = request.args.get('end_date')      # Format: YYYY-MM-DD

        if source:
            df = df[df['source'] == source]

        if year:
            df = df[df['year'] == year]

        if min_quality:
            df = df[df['data_quality_score'] >= min_quality]

        # Filter by date range (conference start date)
        if start_date:
            start_date_ts = pd.to_datetime(start_date)
            df = df[df['conference_start'] >= start_date_ts]

        if end_date:
            end_date_ts = pd.to_datetime(end_date)
            df = df[df['conference_start'] <= end_date_ts]

        # Convert dates to strings for JSON response
        date_columns = ['conference_start', 'conference_end', 'submission_deadline',
                       'notification_date', 'camera_ready_deadline']
        for col in date_columns:
            if col in df.columns:
                df[col] = df[col].dt.strftime('%Y-%m-%d')

        conferences = df.to_dict('records')
        return jsonify(conferences)

    except Exception as e:
        logger.error(f"Error getting conferences: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def get_stats():
    """Get statistics about conferences."""
    try:
        if not os.path.exists(config.CSV_OUTPUT_PATH):
            return jsonify({'error': 'No data available'}), 404

        df = pd.read_csv(config.CSV_OUTPUT_PATH)

        # Parse submission deadlines
        df['submission_deadline'] = pd.to_datetime(df['submission_deadline'], errors='coerce')
        today = pd.Timestamp.now()

        # Calculate statistics
        stats = {
            'total': len(df),
            'by_source': df['source'].value_counts().to_dict(),
            'by_year': df['year'].value_counts().sort_index().to_dict(),
            'avg_quality': float(df['data_quality_score'].mean()),
            'upcoming_deadlines': {
                'within_7_days': len(df[(df['submission_deadline'] >= today) &
                                       (df['submission_deadline'] <= today + timedelta(days=7))]),
                'within_30_days': len(df[(df['submission_deadline'] >= today) &
                                        (df['submission_deadline'] <= today + timedelta(days=30))]),
                'within_90_days': len(df[(df['submission_deadline'] >= today) &
                                        (df['submission_deadline'] <= today + timedelta(days=90))])
            }
        }

        return jsonify(stats)

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/conferences/upcoming')
def get_upcoming():
    """Get conferences with upcoming deadlines."""
    try:
        if not os.path.exists(config.CSV_OUTPUT_PATH):
            return jsonify({'error': 'No data available'}), 404

        df = pd.read_csv(config.CSV_OUTPUT_PATH)
        df['submission_deadline'] = pd.to_datetime(df['submission_deadline'], errors='coerce')

        # Get conferences with upcoming deadlines
        today = pd.Timestamp.now()
        days = request.args.get('days', default=30, type=int)

        upcoming = df[
            (df['submission_deadline'] >= today) &
            (df['submission_deadline'] <= today + timedelta(days=days))
        ].sort_values('submission_deadline')

        # Convert dates to strings
        date_columns = ['conference_start', 'conference_end', 'submission_deadline',
                       'notification_date', 'camera_ready_deadline']
        for col in date_columns:
            if col in upcoming.columns:
                upcoming[col] = upcoming[col].dt.strftime('%Y-%m-%d')

        return jsonify(upcoming.to_dict('records'))

    except Exception as e:
        logger.error(f"Error getting upcoming conferences: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/conferences/search')
def search_conferences():
    """Search conferences by keyword and date range."""
    try:
        if not os.path.exists(config.CSV_OUTPUT_PATH):
            return jsonify({'error': 'No data available'}), 404

        query = request.args.get('q', '').lower()
        start_date = request.args.get('start_date')  # Format: YYYY-MM-DD
        end_date = request.args.get('end_date')      # Format: YYYY-MM-DD

        df = pd.read_csv(config.CSV_OUTPUT_PATH)

        # Parse conference dates
        df['conference_start'] = pd.to_datetime(df['conference_start'], errors='coerce')
        df['conference_end'] = pd.to_datetime(df['conference_end'], errors='coerce')

        # Apply keyword search if query is provided (multiple keywords are OR-combined)
        if query:
            keywords = [kw.strip() for kw in query.split(',') if kw.strip()]
            mask = pd.Series(False, index=df.index)
            for kw in keywords:
                mask = mask | (
                    df['name'].str.lower().str.contains(kw, na=False) |
                    df['topics'].astype(str).str.lower().str.contains(kw, na=False) |
                    df['categories'].astype(str).str.lower().str.contains(kw, na=False) |
                    df['location'].astype(str).str.lower().str.contains(kw, na=False)
                )
            results = df[mask]
        else:
            results = df

        # Apply date range filters
        if start_date:
            start_date_ts = pd.to_datetime(start_date)
            results = results[results['conference_start'] >= start_date_ts]

        if end_date:
            end_date_ts = pd.to_datetime(end_date)
            results = results[results['conference_start'] <= end_date_ts]

        # Sort by conference start date
        results = results.sort_values('conference_start')

        # Convert dates to strings
        date_columns = ['conference_start', 'conference_end', 'submission_deadline',
                       'notification_date', 'camera_ready_deadline']
        for col in date_columns:
            if col in results.columns:
                results[col] = results[col].dt.strftime('%Y-%m-%d')

        return jsonify(results.to_dict('records'))

    except Exception as e:
        logger.error(f"Error searching conferences: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/csv')
def export_csv():
    """Export filtered data as CSV."""
    try:
        if not os.path.exists(config.CSV_OUTPUT_PATH):
            return jsonify({'error': 'No data available'}), 404

        return send_file(
            config.CSV_OUTPUT_PATH,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'conferences_{datetime.now().strftime("%Y%m%d")}.csv'
        )

    except Exception as e:
        logger.error(f"Error exporting CSV: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/excel')
def export_excel():
    """Export filtered data as Excel."""
    try:
        if not os.path.exists(config.EXCEL_OUTPUT_PATH):
            return jsonify({'error': 'No Excel file available'}), 404

        return send_file(
            config.EXCEL_OUTPUT_PATH,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'conferences_{datetime.now().strftime("%Y%m%d")}.xlsx'
        )

    except Exception as e:
        logger.error(f"Error exporting Excel: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/ics')
def export_ics():
    """Export conferences to iCalendar (.ics) format with filtering options."""
    try:
        if not os.path.exists(config.CSV_OUTPUT_PATH):
            return jsonify({'error': 'No data available'}), 404

        df = pd.read_csv(config.CSV_OUTPUT_PATH)

        # Parse conference dates
        df['conference_start'] = pd.to_datetime(df['conference_start'], errors='coerce')
        df['conference_end'] = pd.to_datetime(df['conference_end'], errors='coerce')
        df['submission_deadline'] = pd.to_datetime(df['submission_deadline'], errors='coerce')

        # Apply filters from query parameters
        query = request.args.get('q', '').lower()
        source = request.args.get('source')
        year = request.args.get('year', type=int)
        min_quality = request.args.get('min_quality', type=float)
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        # Filter by keyword
        if query:
            mask = (
                df['name'].str.lower().str.contains(query, na=False) |
                df['topics'].astype(str).str.lower().str.contains(query, na=False) |
                df['categories'].astype(str).str.lower().str.contains(query, na=False) |
                df['location'].astype(str).str.lower().str.contains(query, na=False)
            )
            df = df[mask]

        # Filter by source
        if source:
            df = df[df['source'] == source]

        # Filter by year
        if year:
            df = df[df['year'] == year]

        # Filter by quality
        if min_quality:
            df = df[df['data_quality_score'] >= min_quality]

        # Filter by date range
        if start_date:
            start_date_ts = pd.to_datetime(start_date)
            df = df[df['conference_start'] >= start_date_ts]

        if end_date:
            end_date_ts = pd.to_datetime(end_date)
            df = df[df['conference_start'] <= end_date_ts]

        # Convert to list of dictionaries
        conferences = df.to_dict('records')

        if not conferences:
            return jsonify({'error': 'No conferences match the filters'}), 404

        # Generate calendar name based on filters
        calendar_name = "Academic Conferences"
        if query:
            calendar_name = f"Conferences: {query.title()}"
        elif year:
            calendar_name = f"Conferences {year}"

        # Generate ICS content
        ics_content = ics_exporter.export_to_string(conferences, calendar_name)

        # Return as downloadable file
        filename = f'conferences_{datetime.now().strftime("%Y%m%d")}.ics'

        return Response(
            ics_content,
            mimetype='text/calendar',
            headers={
                'Content-Disposition': f'attachment; filename={filename}',
                'Content-Type': 'text/calendar; charset=utf-8'
            }
        )

    except Exception as e:
        logger.error(f"Error exporting ICS: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    logger.info("Starting Conference Agent Web Dashboard")
    logger.info(f"Data source: {config.CSV_OUTPUT_PATH}")
    app.run(debug=True, host='0.0.0.0', port=5000)
