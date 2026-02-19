#!/bin/bash

# Create the fixed crontab file with environment variables
cat > /home/webapp/fixed_crontab << 'EOF'
APP_DATA_DIR=/var/lib/webapp/data
BACKUP_DIR=/var/backups/webapp
REMOTE_PATH=s3://company-backups/webapp
STORAGE_CLI=/usr/local/bin/s3cmd
RETENTION_DAYS=7
0 3 * * * /opt/webapp/backup.sh
EOF

# Set proper ownership
chown webapp:webapp /home/webapp/fixed_crontab