# Gunicorn configuration file
# File: /opt/webapp/gunicorn_config.py
# This configuration demonstrates the timeout problem

import multiprocessing
import os

# Server socket binding
bind = '0.0.0.0:8000'

# Worker processes
workers = 4

# Worker class - sync workers for CPU-bound tasks
worker_class = 'sync'

# Worker timeout - THIS IS THE PROBLEM
# 30 seconds is too short for CSV processing operations that take 90-120 seconds
timeout = 30

# Worker connections (for async workers)
worker_connections = 1000

# Maximum number of requests a worker will process before restarting
max_requests = 1000
max_requests_jitter = 50

# Workers silent for more than this many seconds are killed and restarted
# This also affects long-running operations
graceful_timeout = 30

# Timeout for graceful workers restart
keepalive = 5

# Logging
accesslog = '/opt/webapp/access.log'
errorlog = '/opt/webapp/error.log'
loglevel = 'info'

# Log format
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = 'webapp'

# Daemon mode
daemon = False

# User and group to run workers as
user = None
group = None

# Directory to change to before loading apps
chdir = '/opt/webapp'

# Preload application code before worker processes are forked
preload_app = False

# Server mechanics
pidfile = '/opt/webapp/gunicorn.pid'
umask = 0
tmp_upload_dir = None