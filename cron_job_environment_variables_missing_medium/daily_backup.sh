#!/bin/bash
# Backup script for database and remote storage
# Must be executable: chmod +x /opt/backup/daily_backup.sh

set -e

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a /var/log/backup.log
}

log "Starting backup process..."

# Check required environment variables
if [ -z "$DATABASE_URL" ]; then
    log "ERROR: DATABASE_URL not set"
    exit 1
fi

if [ -z "$DB_USER" ]; then
    log "ERROR: DB_USER not set"
    exit 1
fi

if [ -z "$DB_PASSWORD" ]; then
    log "ERROR: DB_PASSWORD not set"
    exit 1
fi

if [ -z "$API_KEY" ]; then
    log "ERROR: API_KEY not set"
    exit 1
fi

if [ -z "$BACKUP_PATH" ]; then
    log "ERROR: BACKUP_PATH not set"
    exit 1
fi

# Verify PATH is set
if [ -z "$PATH" ]; then
    log "ERROR: PATH not set"
    exit 1
fi

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_PATH"
log "Backup directory: $BACKUP_PATH"

# Generate backup filename with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="backup_${TIMESTAMP}.sql"
BACKUP_ARCHIVE="backup_${TIMESTAMP}.tar.gz"

# Dump database
log "Dumping database from $DATABASE_URL"
mysqldump -h "$DATABASE_URL" -u "$DB_USER" -p"$DB_PASSWORD" --all-databases > "${BACKUP_PATH}/${BACKUP_FILE}"

if [ $? -ne 0 ]; then
    log "ERROR: Database dump failed"
    exit 1
fi

log "Database dump completed: ${BACKUP_FILE}"

# Compress the backup
log "Compressing backup..."
cd "$BACKUP_PATH"
tar -czf "$BACKUP_ARCHIVE" "$BACKUP_FILE"

if [ $? -ne 0 ]; then
    log "ERROR: Compression failed"
    exit 1
fi

log "Backup compressed: ${BACKUP_ARCHIVE}"

# Upload to remote storage
log "Uploading backup to remote storage..."
curl -X POST \
    -H "Authorization: Bearer $API_KEY" \
    -F "file=@${BACKUP_PATH}/${BACKUP_ARCHIVE}" \
    https://storage.example.com/api/upload

if [ $? -ne 0 ]; then
    log "ERROR: Upload to remote storage failed"
    exit 1
fi

log "Backup uploaded successfully"

# Clean up old backups (keep last 7 days)
log "Cleaning up old backups..."
find "$BACKUP_PATH" -name "backup_*.tar.gz" -mtime +7 -delete
find "$BACKUP_PATH" -name "backup_*.sql" -delete

log "Backup process completed successfully"
exit 0