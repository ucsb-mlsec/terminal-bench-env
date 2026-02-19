#!/bin/bash

# Current Production MySQL Backup Script
# WARNING: This script has been failing with lock wait timeout errors
# for the past week during nightly backup runs

# Database configuration
DB_NAME='ecommerce_prod'
DB_USER='backup_user'
DB_PASS='backup_pass123'
BACKUP_DIR='/backups/mysql'

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Generate timestamp for backup filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/backup_${TIMESTAMP}.sql"

echo "Starting backup at $(date)..."

# Set lock wait timeout to 50 seconds (too short for high-traffic production DB)
mysql -u"$DB_USER" -p"$DB_PASS" -e "SET SESSION lock_wait_timeout=50;"

# Run mysqldump with problematic options
# --lock-tables causes locks on all tables which conflicts with ongoing transactions
# --max-allowed-packet=128M may be too small for large tables
mysqldump -u"$DB_USER" -p"$DB_PASS" \
  --lock-tables \
  --max-allowed-packet=128M \
  --databases "$DB_NAME" \
  > "$BACKUP_FILE" 2>&1

# Check if backup succeeded
if [ $? -eq 0 ]; then
    echo "Backup completed successfully: $BACKUP_FILE"
else
    echo "Backup failed! Check error logs."
    exit 1
fi