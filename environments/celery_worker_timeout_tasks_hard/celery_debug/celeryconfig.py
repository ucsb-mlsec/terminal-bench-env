#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Celery Configuration File

# Broker settings
broker_url = 'redis://localhost:6379/0'
result_backend = 'redis://localhost:6379/0'

# Serialization settings
task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']

# Timezone settings
timezone = 'UTC'
enable_utc = True

# Task execution time limits
task_soft_time_limit = 120  # seconds - soft time limit (warning)
task_time_limit = 150  # seconds - hard time limit (forced termination)

# Worker settings
worker_prefetch_multiplier = 4

# Task acknowledgment settings
task_acks_late = True
task_reject_on_worker_lost = True