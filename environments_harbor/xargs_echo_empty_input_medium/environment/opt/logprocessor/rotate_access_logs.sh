#!/bin/bash

# Archive access logs older than 60 days
# This script compresses old access log files to save disk space

LOG_DIR="/var/app/logs"
ARCHIVE_DIR="/var/app/archive"
ARCHIVE_LOG="/var/app/logs/archive.log"

# Find and archive old access logs
find /var/app/logs/ -name "access.log.*" -mtime +60 | xargs tar -czf /var/app/archive/access_$(date +%Y%m%d).tar.gz

# Log the archival operation
echo "Archived access logs at $(date '+%Y-%m-%d %H:%M:%S')" >> /var/app/logs/archive.log