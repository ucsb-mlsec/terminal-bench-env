#!/usr/bin/env python3

import os
import sys
import time
import yaml
import logging
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.pool import ThreadPoolExecutor
import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from YAML file"""
    config_path = os.getenv('SCHEDULER_CONFIG', '/etc/scheduler/config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_db_connection(config):
    """Create database connection"""
    return psycopg2.connect(
        host=config['database']['host'],
        port=config['database']['port'],
        database=config['database']['name'],
        user=config['database']['user'],
        password=config['database']['password']
    )


def cleanup_old_records():
    """Scheduled job: Clean up old records from database"""
    logger.info("Running cleanup job for old records")
    try:
        config = load_config()
        conn = get_db_connection(config)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM logs WHERE created_at < NOW() - INTERVAL '30 days'")
        deleted = cursor.rowcount
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"Cleanup completed: {deleted} records deleted")
    except Exception as e:
        logger.error(f"Cleanup job failed: {e}")


def generate_daily_report():
    """Scheduled job: Generate daily analytics report"""
    logger.info("Generating daily report")
    try:
        config = load_config()
        conn = get_db_connection(config)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT COUNT(*) as total, status FROM tasks GROUP BY status")
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        logger.info(f"Daily report generated: {results}")
    except Exception as e:
        logger.error(f"Report generation failed: {e}")


def sync_external_data():
    """Scheduled job: Sync data from external API"""
    logger.info("Syncing external data")
    try:
        config = load_config()
        # Simulate API call and data sync
        time.sleep(2)
        logger.info("External data sync completed successfully")
    except Exception as e:
        logger.error(f"Data sync failed: {e}")


def health_check():
    """Scheduled job: Perform system health check"""
    logger.info("Performing health check")
    try:
        config = load_config()
        conn = get_db_connection(config)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        logger.info("Health check passed")
    except Exception as e:
        logger.error(f"Health check failed: {e}")


def main():
    """Main scheduler setup and execution"""
    logger.info("Starting scheduler service")
    
    config = load_config()
    
    # Configure job stores and executors
    jobstores = {
        'default': SQLAlchemyJobStore(url=config['scheduler']['jobstore_url'])
    }
    executors = {
        'default': ThreadPoolExecutor(max_workers=config['scheduler'].get('max_workers', 10))
    }
    job_defaults = {
        'coalesce': False,
        'max_instances': 3
    }
    
    scheduler = BackgroundScheduler(
        jobstores=jobstores,
        executors=executors,
        job_defaults=job_defaults
    )
    
    # Schedule jobs with different intervals
    scheduler.add_job(cleanup_old_records, 'interval', hours=24, id='cleanup_job')
    scheduler.add_job(generate_daily_report, 'cron', hour=8, minute=0, id='daily_report')
    scheduler.add_job(sync_external_data, 'interval', minutes=15, id='data_sync')
    scheduler.add_job(health_check, 'interval', minutes=5, id='health_check')
    
    scheduler.start()
    logger.info("Scheduler started successfully")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down scheduler")
        scheduler.shutdown()
        sys.exit(0)


if __name__ == '__main__':
    main()