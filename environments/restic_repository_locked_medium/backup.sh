#!/bin/bash

# Backup script with lock file mechanism to prevent concurrent backups
# This script backs up /srv/application/data to /backup/archive

LOCKFILE=/var/lock/backup.lock
BACKUP_SOURCE=/srv/application/data
BACKUP_DEST=/backup/archive
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_FILE="${BACKUP_DEST}/backup-${TIMESTAMP}.tar.gz"

echo "Starting backup process at $(date)"

# Check if lock file exists
if [ -f "$LOCKFILE" ]; then
    echo "Lock file found at $LOCKFILE"
    
    # Read the PID from the lock file
    LOCKED_PID=$(cat "$LOCKFILE")
    echo "Lock file contains PID: $LOCKED_PID"
    
    # Check if the process is actually running
    if ps -p "$LOCKED_PID" > /dev/null 2>&1; then
        echo "ERROR: Backup process with PID $LOCKED_PID is already running"
        echo "Cannot start backup while another backup is in progress"
        exit 1
    else
        # Bug: Script refuses to run even when process is NOT running
        echo "ERROR: Lock file exists but process $LOCKED_PID is not running"
        echo "This appears to be a stale lock file"
        echo "Refusing to run - manual intervention required"
        exit 1
    fi
fi

# Create lock file with current PID
echo "Creating lock file with PID $$"
echo $$ > "$LOCKFILE"

# Verify lock file was created
if [ ! -f "$LOCKFILE" ]; then
    echo "ERROR: Failed to create lock file"
    exit 1
fi

# Perform the backup
echo "Creating backup of $BACKUP_SOURCE"
echo "Destination: $BACKUP_FILE"

tar -czf "$BACKUP_FILE" -C "$(dirname $BACKUP_SOURCE)" "$(basename $BACKUP_SOURCE)" 2>/dev/null

# Check if backup was successful
if [ $? -eq 0 ] && [ -f "$BACKUP_FILE" ]; then
    echo "Backup completed successfully"
    echo "Backup file: $BACKUP_FILE"
    echo "Size: $(du -h "$BACKUP_FILE" | cut -f1)"
else
    echo "ERROR: Backup failed"
    rm -f "$LOCKFILE"
    exit 1
fi

# Remove lock file
echo "Removing lock file"
rm -f "$LOCKFILE"

echo "Backup process completed at $(date)"
exit 0