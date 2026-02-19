#!/bin/bash

# Sync transaction files to backup server
# This script handles backup synchronization of transaction data
# Already uses -r flag correctly

echo "Syncing files to backup system"

# Sync recent transaction files to backup server
# Using -r flag to skip empty input
find /var/data/transactions/ -name "*.txn" -mtime -1 | xargs -r -n 100 rsync -av --files-from=- /var/data/transactions/ backup-server:/backup/

# Backup critical data files
# Already uses -r flag correctly
ls /var/data/transactions/critical_*.dat 2>/dev/null | xargs -r cp -t /mnt/backup/

# Log completion
echo "Backup sync completed at $(date)"