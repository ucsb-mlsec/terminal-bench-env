#!/bin/bash

# Log Rotation Utility Script
# Provides operations for managing, compressing, archiving, and validating log files

usage() {
    cat << EOF
Usage: $0 <operation> [arguments]

Operations:
  compress <log_file>              Compress a log file using gzip
  archive <log_file> <backup_dir>  Move log file to backup directory
  cleanup <directory> <days>       Delete log files older than specified days
  validate <log_file>              Validate log file format and content
  stats <directory>                Show statistics about log files in directory

Examples:
  $0 compress /var/log/app.log
  $0 archive /var/log/app.log /backup/logs
  $0 cleanup /var/log 30
  $0 validate /var/log/app.log
  $0 stats /var/log

Exit Codes:
  0 - Success
  1 - Error
EOF
}

# COMPRESS operation: Compress log file using gzip
compress_log() {
    local log_file="$1"
    
    if [ -z "$log_file" ]; then
        echo "Error: Log file path required" >&2
        return 1
    fi
    
    if [ ! -f "$log_file" ]; then
        echo "Error: File '$log_file' does not exist" >&2
        return 1
    fi
    
    if [ ! -r "$log_file" ]; then
        echo "Error: File '$log_file' is not readable" >&2
        return 1
    fi
    
    # Compress the file
    if gzip "$log_file" 2>/dev/null; then
        echo "Successfully compressed '$log_file' to '${log_file}.gz'"
        return 0
    else
        echo "Error: Failed to compress '$log_file'" >&2
        return 1
    fi
}

# ARCHIVE operation: Move log file to backup directory
archive_log() {
    local log_file="$1"
    local backup_dir="$2"
    
    if [ -z "$log_file" ]; then
        echo "Error: Log file path required" >&2
        return 1
    fi
    
    if [ -z "$backup_dir" ]; then
        echo "Error: Backup directory required" >&2
        return 1
    fi
    
    if [ ! -f "$log_file" ]; then
        echo "Error: File '$log_file' does not exist" >&2
        return 1
    fi
    
    # Create backup directory if it doesn't exist
    if [ ! -d "$backup_dir" ]; then
        if ! mkdir -p "$backup_dir" 2>/dev/null; then
            echo "Error: Failed to create backup directory '$backup_dir'" >&2
            return 1
        fi
    fi
    
    # Move the file to backup directory
    local filename=$(basename "$log_file")
    if mv "$log_file" "$backup_dir/$filename" 2>/dev/null; then
        echo "Successfully archived '$log_file' to '$backup_dir/$filename'"
        return 0
    else
        echo "Error: Failed to archive '$log_file'" >&2
        return 1
    fi
}

# CLEANUP operation: Delete log files older than specified days
cleanup_logs() {
    local directory="$1"
    local days="$2"
    
    if [ -z "$directory" ]; then
        echo "Error: Directory path required" >&2
        return 1
    fi
    
    if [ -z "$days" ]; then
        echo "Error: Number of days required" >&2
        return 1
    fi
    
    if [ ! -d "$directory" ]; then
        echo "Error: Directory '$directory' does not exist" >&2
        return 1
    fi
    
    # Validate days is a number
    if ! [[ "$days" =~ ^[0-9]+$ ]]; then
        echo "Error: Days must be a positive number" >&2
        return 1
    fi
    
    # Find and count old log files
    local count=0
    while IFS= read -r file; do
        if [ -f "$file" ]; then
            rm -f "$file" 2>/dev/null && ((count++))
        fi
    done < <(find "$directory" -name "*.log" -type f -mtime +"$days" 2>/dev/null)
    
    echo "Deleted $count log file(s) older than $days days"
    return 0
}

# VALIDATE operation: Check if log file has valid format
validate_log() {
    local log_file="$1"
    
    if [ -z "$log_file" ]; then
        echo "Error: Log file path required" >&2
        return 1
    fi
    
    # Check if file exists
    if [ ! -f "$log_file" ]; then
        echo "Validation failed: File '$log_file' does not exist" >&2
        return 1
    fi
    
    # Check if file has .log extension
    if [[ ! "$log_file" =~ \.log$ ]]; then
        echo "Validation failed: File '$log_file' does not have .log extension" >&2
        return 1
    fi
    
    # Check if file is readable
    if [ ! -r "$log_file" ]; then
        echo "Validation failed: File '$log_file' is not readable" >&2
        return 1
    fi
    
    # Check if file is non-empty
    if [ ! -s "$log_file" ]; then
        echo "Validation failed: File '$log_file' is empty" >&2
        return 1
    fi
    
    echo "Validation successful: '$log_file' is a valid log file"
    return 0
}

# STATS operation: Report statistics about log files in directory
stats_logs() {
    local directory="$1"
    
    if [ -z "$directory" ]; then
        echo "Error: Directory path required" >&2
        return 1
    fi
    
    if [ ! -d "$directory" ]; then
        echo "Error: Directory '$directory' does not exist" >&2
        return 1
    fi
    
    # Count total log files
    local total_files=$(find "$directory" -name "*.log" -type f 2>/dev/null | wc -l)
    
    if [ "$total_files" -eq 0 ]; then
        echo "Log Statistics for '$directory':"
        echo "  Total log files: 0"
        echo "  Total size: 0 bytes"
        echo "  Oldest file: N/A"
        echo "  Newest file: N/A"
        return 0
    fi
    
    # Calculate total size
    local total_size=0
    while IFS= read -r file; do
        if [ -f "$file" ]; then
            local size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo 0)
            total_size=$((total_size + size))
        fi
    done < <(find "$directory" -name "*.log" -type f 2>/dev/null)
    
    # Find oldest file
    local oldest_file=$(find "$directory" -name "*.log" -type f -printf '%T+ %p\n' 2>/dev/null | sort | head -n1 | cut -d' ' -f2-)
    if [ -z "$oldest_file" ]; then
        oldest_file=$(find "$directory" -name "*.log" -type f -print0 2>/dev/null | xargs -0 ls -t | tail -n1)
    fi
    
    # Find newest file
    local newest_file=$(find "$directory" -name "*.log" -type f -printf '%T+ %p\n' 2>/dev/null | sort -r | head -n1 | cut -d' ' -f2-)
    if [ -z "$newest_file" ]; then
        newest_file=$(find "$directory" -name "*.log" -type f -print0 2>/dev/null | xargs -0 ls -t | head -n1)
    fi
    
    # Output statistics
    echo "Log Statistics for '$directory':"
    echo "  Total log files: $total_files"
    echo "  Total size: $total_size bytes"
    echo "  Oldest file: ${oldest_file:-N/A}"
    echo "  Newest file: ${newest_file:-N/A}"
    
    return 0
}

# Main script logic
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

operation="$1"
shift

case "$operation" in
    compress)
        compress_log "$@"
        exit $?
        ;;
    archive)
        archive_log "$@"
        exit $?
        ;;
    cleanup)
        cleanup_logs "$@"
        exit $?
        ;;
    validate)
        validate_log "$@"
        exit $?
        ;;
    stats)
        stats_logs "$@"
        exit $?
        ;;
    *)
        echo "Error: Invalid operation '$operation'" >&2
        echo ""
        usage
        exit 1
        ;;
esac