#!/bin/bash

# Database Backup Script
# This script performs automated backups of the production database

BACKUP_DIR="/var/backups/database"
LOG_FILE="/var/log/database-backup.log"
DB_NAME="production_db"
DB_USER="backup_user"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
BACKUP_FILE="${BACKUP_DIR}/backup-${TIMESTAMP}.sql.gz"

# Create backup directory if it doesn't exist
mkdir -p "${BACKUP_DIR}"

# Log start of backup
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting database backup..." >> "${LOG_FILE}"

# Perform database backup with mysqldump
if command -v mysqldump &> /dev/null; then
    mysqldump -u "${DB_USER}" "${DB_NAME}" 2>>"${LOG_FILE}" | gzip > "${BACKUP_FILE}"
    RESULT=$?
elif command -v pg_dump &> /dev/null; then
    pg_dump -U "${DB_USER}" "${DB_NAME}" 2>>"${LOG_FILE}" | gzip > "${BACKUP_FILE}"
    RESULT=$?
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: No database dump utility found" >> "${LOG_FILE}"
    exit 1
fi

# Check if backup was successful
if [ $RESULT -eq 0 ] && [ -f "${BACKUP_FILE}" ]; then
    BACKUP_SIZE=$(du -h "${BACKUP_FILE}" | cut -f1)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Backup completed successfully: ${BACKUP_FILE} (${BACKUP_SIZE})" >> "${LOG_FILE}"
    
    # Remove backups older than 30 days
    find "${BACKUP_DIR}" -name "backup-*.sql.gz" -mtime +30 -delete 2>>"${LOG_FILE}"
    
    exit 0
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Backup failed" >> "${LOG_FILE}"
    exit 1
fi