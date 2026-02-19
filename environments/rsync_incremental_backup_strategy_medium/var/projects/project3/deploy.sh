#!/bin/bash

# Backup script with incremental backup strategy using hard links

SOURCE_DIR="/var/projects"
BACKUP_BASE="/backup/storage"
RETENTION_DAYS=7
TIMESTAMP=$(date +"%Y-%m-%d-%H%M%S")
CURRENT_BACKUP="${BACKUP_BASE}/backup-${TIMESTAMP}"

# Create backup base directory if it doesn't exist
mkdir -p "${BACKUP_BASE}"

# Find the most recent backup for hard linking
LATEST_BACKUP=$(find "${BACKUP_BASE}" -maxdepth 1 -type d -name "backup-*" | sort | tail -n 1)

# Perform the backup using rsync
if [ -n "${LATEST_BACKUP}" ] && [ -d "${LATEST_BACKUP}" ]; then
    # Incremental backup with hard links to previous backup
    rsync -a --delete --link-dest="${LATEST_BACKUP}" "${SOURCE_DIR}/" "${CURRENT_BACKUP}/"
else
    # First full backup
    rsync -a --delete "${SOURCE_DIR}/" "${CURRENT_BACKUP}/"
fi

# Rotate old backups - remove snapshots older than RETENTION_DAYS
find "${BACKUP_BASE}" -maxdepth 1 -type d -name "backup-*" -mtime +${RETENTION_DAYS} -exec rm -rf {} \;

exit 0
```

Now creating the backup_info.txt file:

```bash
#!/bin/bash

cat > /root/backup_info.txt << 'EOF'
SCRIPT_PATH=/root/backup_solution.sh
BACKUP_DESTINATION=/backup/storage
RETENTION_DAYS=7
EOF
```

Now creating the deploy.sh file in the projects directory:

```bash
#!/bin/bash

set -e

# Deployment Configuration
APP_NAME="myapp"
DEPLOY_DIR="/opt/${APP_NAME}"
REPO_URL="https://github.com/company/myapp.git"
BUILD_DIR="/tmp/build-${APP_NAME}"
BACKUP_DIR="/var/backups/${APP_NAME}"
LOG_FILE="/var/log/${APP_NAME}-deploy.log"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

# Function to create backup
backup_current() {
    log_message "Creating backup of current deployment..."
    if [ -d "${DEPLOY_DIR}" ]; then
        mkdir -p "${BACKUP_DIR}"
        cp -r "${DEPLOY_DIR}" "${BACKUP_DIR}/backup-$(date +%Y%m%d-%H%M%S)"
        log_message "Backup created successfully"
    fi
}

# Function to build application
build_app() {
    log_message "Starting build process..."
    cd "${BUILD_DIR}"
    git pull origin main
    npm install --production
    npm run build
    log_message "Build completed successfully"
}

# Function to run tests
run_tests() {
    log_message "Running tests..."
    cd "${BUILD_DIR}"
    npm test
    if [ $? -eq 0 ]; then
        log_message "All tests passed"
    else
        log_message "Tests failed"
        exit 1
    fi
}

# Function to deploy application
deploy_app() {
    log_message "Deploying application..."
    mkdir -p "${DEPLOY_DIR}"
    cp -r "${BUILD_DIR}/dist/"* "${DEPLOY_DIR}/"
    docker-compose -f "${DEPLOY_DIR}/docker-compose.yml" up -d
    log_message "Deployment completed"
}

# Function to cleanup
cleanup() {
    log_message "Cleaning up temporary files..."
    rm -rf "${BUILD_DIR}"
    log_message "Cleanup completed"
}

# Main deployment workflow
log_message "Starting deployment for ${APP_NAME}"

backup_current
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

if [ ! -d "${BUILD_DIR}/.git" ]; then
    git clone "${REPO_URL}" "${BUILD_DIR}"
fi

build_app
run_tests
deploy_app
cleanup

log_message "Deployment finished successfully"
exit 0