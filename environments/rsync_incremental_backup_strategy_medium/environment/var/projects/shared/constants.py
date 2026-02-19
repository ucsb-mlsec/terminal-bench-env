#!/bin/bash

# Backup script for incremental backups with hard-linking
# Author: System Administrator
# Description: Creates incremental backups of /var/projects to /backup/storage
#              with automatic rotation of backups older than 7 days

set -e

# Configuration
SOURCE_DIR="/var/projects"
BACKUP_BASE="/backup/storage"
RETENTION_DAYS=7
TIMESTAMP=$(date +%Y-%m-%d-%H%M%S)
BACKUP_NAME="backup-${TIMESTAMP}"
BACKUP_DEST="${BACKUP_BASE}/${BACKUP_NAME}"

# Create backup base directory if it doesn't exist
mkdir -p "${BACKUP_BASE}"

# Find the most recent backup for hard-linking
LATEST_BACKUP=$(find "${BACKUP_BASE}" -maxdepth 1 -type d -name "backup-*" | sort -r | head -n 1)

# Perform the backup
if [ -z "${LATEST_BACKUP}" ]; then
    # First backup - no hard-linking reference
    echo "Performing full backup to ${BACKUP_DEST}"
    rsync -a --delete \
        "${SOURCE_DIR}/" \
        "${BACKUP_DEST}/"
else
    # Incremental backup with hard-linking
    echo "Performing incremental backup to ${BACKUP_DEST}"
    echo "Using reference: ${LATEST_BACKUP}"
    rsync -a --delete \
        --link-dest="${LATEST_BACKUP}" \
        "${SOURCE_DIR}/" \
        "${BACKUP_DEST}/"
fi

# Rotate old backups (remove backups older than RETENTION_DAYS)
echo "Rotating old backups (removing backups older than ${RETENTION_DAYS} days)"
find "${BACKUP_BASE}" -maxdepth 1 -type d -name "backup-*" -mtime +${RETENTION_DAYS} -exec rm -rf {} \;

# Count remaining backups
BACKUP_COUNT=$(find "${BACKUP_BASE}" -maxdepth 1 -type d -name "backup-*" | wc -l)
echo "Backup completed successfully: ${BACKUP_DEST}"
echo "Total backup snapshots: ${BACKUP_COUNT}"

exit 0
```

```python
"""
Constants Module

This module contains application-wide constants used throughout the project.
All constants follow UPPER_CASE naming convention and should not be modified
at runtime. These include API configurations, system limits, file handling
parameters, status codes, and path definitions.

Author: Development Team
Version: 1.0.0
"""

# API Configuration Constants
API_VERSION = "v2.1.0"
DEFAULT_ENCODING = "utf-8"
APP_NAME = "ProjectBackupSystem"
BASE_URL = "https://api.example.com"
API_KEY_HEADER = "X-API-Key"

# Numeric Configuration Constants
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30
BUFFER_SIZE = 8192
MAX_FILE_SIZE_MB = 100
CONNECTION_POOL_SIZE = 10
DEFAULT_PORT = 8080
MAX_CONNECTIONS = 50

# List and Tuple Constants
ALLOWED_EXTENSIONS = ('.txt', '.py', '.json', '.xml', '.csv', '.log')
VALID_STATUSES = ('pending', 'processing', 'completed', 'failed', 'cancelled')
ERROR_CODES = (400, 401, 403, 404, 500, 502, 503)
SUPPORTED_FORMATS = ['json', 'xml', 'yaml', 'csv']

# Dictionary Constants
STATUS_MESSAGES = {
    'pending': 'Operation is pending',
    'processing': 'Operation is in progress',
    'completed': 'Operation completed successfully',
    'failed': 'Operation failed',
    'cancelled': 'Operation was cancelled'
}

ERROR_TEMPLATES = {
    400: 'Bad Request: {message}',
    401: 'Unauthorized: {message}',
    403: 'Forbidden: {message}',
    404: 'Not Found: {message}',
    500: 'Internal Server Error: {message}'
}

# Path Constants
DEFAULT_LOG_PATH = "/var/log/projects/application.log"
CONFIG_FILE_PATH = "/etc/projects/config.ini"
TEMP_DIRECTORY = "/tmp/projects"
DATA_DIRECTORY = "/var/projects/data"
```

```bash
cat > /root/backup_info.txt << 'EOF'
SCRIPT_PATH=/root/backup_solution.sh
BACKUP_DESTINATION=/backup/storage
RETENTION_DAYS=7
EOF