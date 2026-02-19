#!/usr/bin/env python3

import os
import signal
import sys
import logging
from celery import Celery, Task
from celery.signals import worker_shutdown, worker_ready
import redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost/dbname')
BROKER_URL = os.getenv('BROKER_URL', REDIS_URL)

# Initialize Celery app
app = Celery('worker', broker=BROKER_URL, backend=REDIS_URL)
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,
)

# Database setup
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(bind=engine)

# Redis connection
redis_client = redis.from_url(REDIS_URL)


@app.task(bind=True, name='process_data')
def process_data(self, data_id):
    """Process data from database"""
    logger.info(f"Processing data with ID: {data_id}")
    db = SessionLocal()
    try:
        # Simulate data processing
        result = {'data_id': data_id, 'status': 'processed'}
        logger.info(f"Data {data_id} processed successfully")
        return result
    except Exception as e:
        logger.error(f"Error processing data {data_id}: {str(e)}")
        raise self.retry(exc=e, countdown=60, max_retries=3)
    finally:
        db.close()


@app.task(bind=True, name='send_notifications')
def send_notifications(self, user_ids, message):
    """Send notifications to multiple users"""
    logger.info(f"Sending notifications to {len(user_ids)} users")
    try:
        for user_id in user_ids:
            # Simulate notification sending
            redis_client.lpush(f'notifications:{user_id}', message)
        return {'sent': len(user_ids), 'message': message}
    except Exception as e:
        logger.error(f"Error sending notifications: {str(e)}")
        raise self.retry(exc=e, countdown=30, max_retries=5)


@app.task(bind=True, name='generate_report')
def generate_report(self, report_type, params):
    """Generate reports based on type and parameters"""
    logger.info(f"Generating {report_type} report with params: {params}")
    db = SessionLocal()
    try:
        # Simulate report generation
        report_data = {'type': report_type, 'params': params, 'generated': True}
        redis_client.setex(f'report:{report_type}', 3600, str(report_data))
        return report_data
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise self.retry(exc=e, countdown=120, max_retries=2)
    finally:
        db.close()


@app.task(name='cleanup_old_data')
def cleanup_old_data(days=30):
    """Clean up old data from database and cache"""
    logger.info(f"Cleaning up data older than {days} days")
    db = SessionLocal()
    try:
        # Simulate cleanup
        cleaned_count = 0
        logger.info(f"Cleaned up {cleaned_count} records")
        return {'cleaned': cleaned_count, 'days': days}
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        raise
    finally:
        db.close()


@worker_ready.connect
def on_worker_ready(sender, **kwargs):
    """Handle worker ready signal"""
    logger.info("Worker is ready and waiting for tasks")


@worker_shutdown.connect
def on_worker_shutdown(sender, **kwargs):
    """Handle worker shutdown signal"""
    logger.info("Worker shutting down gracefully")
    engine.dispose()
    redis_client.close()


def signal_handler(signum, frame):
    """Handle termination signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    logger.info("Starting Celery worker...")
    app.start()