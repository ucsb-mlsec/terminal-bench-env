#!/bin/bash

# LVM Snapshot Backup Script
# Purpose: Create snapshot of database volume and backup

# Configuration variables
VOLUME_GROUP='vg_data'
LOGICAL_VOLUME='lv_database'
SNAPSHOT_NAME='snap_backup'
SNAPSHOT_SIZE='1G'
MOUNT_POINT='/mnt/backup_snap'
BACKUP_DIR='/backup'
BACKUP_FILE='db_backup.tar.gz'

# Create snapshot
lvcreate -L ${SNAPSHOT_SIZE} -s -n ${SNAPSHOT_NAME} /dev/${VOLUME_GROUP}/${LOGICAL_VOLUME}

# Mount the snapshot
mount /dev/${VOLUME_GROUP}/${SNAPSHOT_NAME} ${MOUNT_POINT}

# Backup database
tar -czf ${BACKUP_DIR}/${BACKUP_FILE} ${MOUNT_POINT}

# Unmount snapshot
umount ${MOUNT_POINT}

# Remove snapshot
lvremove -f /dev/${VOLUME_GROUP}/${SNAPSHOT_NAME}