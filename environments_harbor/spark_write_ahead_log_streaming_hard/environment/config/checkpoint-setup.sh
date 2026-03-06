#!/bin/bash
# Checkpoint Directory Setup Script

CHECKPOINT_DIR="/checkpoint"
WAL_DIR="${CHECKPOINT_DIR}/receivedData"

# Create checkpoint directory
mkdir -p ${CHECKPOINT_DIR}

# Create WAL directory
mkdir -p ${WAL_DIR}

# Set permissions (INCORRECT - too restrictive for WAL operations)
chmod 644 ${CHECKPOINT_DIR}
chmod 644 ${WAL_DIR}

# Set ownership
chown -R spark:spark ${CHECKPOINT_DIR}

echo "Checkpoint directories created"
echo "Directory: ${CHECKPOINT_DIR}"
echo "WAL Directory: ${WAL_DIR}"

# Note: Permissions should be 755 or 775 for proper WAL write access