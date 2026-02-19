#!/usr/bin/env python
# -*- coding: utf-8 -*-

from celery import Celery
import redis

# Initialize Celery app
app = Celery('image_processor')

# Configure Redis broker
app.conf.broker_url = 'redis://localhost:6379/0'

# Celery configuration
app.conf.update(
    result_backend='redis://localhost:6379/0',
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=4,
    task_time_limit=300,
    task_soft_time_limit=240,
)

# Global cache that accumulates task metadata without clearing
# This dictionary grows indefinitely as tasks are processed
task_metadata_cache = {}

def cache_task_metadata(task_id, metadata):
    """Store task metadata in global cache - never clears, causes memory leak"""
    task_metadata_cache[task_id] = {
        'timestamp': metadata.get('timestamp'),
        'task_name': metadata.get('task_name'),
        'args': metadata.get('args'),
        'result': metadata.get('result'),
    }

# Import task modules to register tasks with Celery
from tasks import process_image, generate_thumbnail, batch_process_images
