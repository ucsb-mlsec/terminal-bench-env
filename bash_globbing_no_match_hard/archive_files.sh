#!/bin/bash

# Archive Files Script
# Archives processed files to dated backup directories

PROCESSED_DIR="/home/agent/processed"
ARCHIVE_BASE="/home/agent/archive"
TEMP_DIR="/home/agent/temp"

# Create archive directory if it doesn't exist
mkdir -p "$ARCHIVE_BASE"

# Archive text files to date-based directories
echo "Archiving text files..."
for file in $PROCESSED_DIR/*.txt; do
    filename=$(basename "$file")
    echo "Moving $filename to archive"
    mv "$file" "$ARCHIVE_BASE/"
done

# Count backup files in all archive subdirectories
echo "Counting backup files..."
count=$(/bin/ls $ARCHIVE_BASE/*/*.bak | wc -l)
echo "Found $count backup files"

# Create compressed archive of data files
echo "Creating compressed archive..."
tar -czf $ARCHIVE_BASE/backup.tar.gz $PROCESSED_DIR/*.dat

# Clean up temporary files
echo "Cleaning temporary files..."
rm $TEMP_DIR/*.tmp
rm $TEMP_DIR/*.cache

# Process archive directories with date patterns
echo "Processing dated archives..."
for dir in $ARCHIVE_BASE/2024-01-*/; do
    echo "Processing archive directory: $dir"
    file_count=$(ls "$dir"*.log 2>/dev/null | wc -l)
    echo "Files in $dir: $file_count"
done

# Validate archived files
echo "Validating archived files..."
archived_files=($ARCHIVE_BASE/*.txt)
if [ ${#archived_files[@]} -gt 0 ]; then
    echo "Archive validation: ${#archived_files[@]} files found"
fi

# Remove old backup files from dated directories
echo "Removing old backups..."
for backup in $ARCHIVE_BASE/2024-*/*.bak; do
    rm "$backup"
done

# Summary of archive operations
echo "Archive operations completed"