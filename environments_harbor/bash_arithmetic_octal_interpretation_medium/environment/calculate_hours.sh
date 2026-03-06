#!/bin/bash

# Legacy time tracking calculation script
# WARNING: Contains known bugs with certain date formats

DATE=$1

if [ -z "$DATE" ]; then
    echo "Error: Date argument required in YYYY-MM-DD format"
    exit 1
fi

LOG_FILE="/workspace/timetracker/logs/${DATE}.log"
OUTPUT_FILE="/workspace/timetracker/results.txt"

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: Log file not found: $LOG_FILE"
    exit 1
fi

# Clear previous results
> "$OUTPUT_FILE"

# Process each log entry
while IFS=',' read -r EMPLOYEE_ID CLOCK_IN CLOCK_OUT; do
    # Extract hours and minutes from HHMM format
    IN_HOUR=$(( CLOCK_IN / 100 ))
    IN_MIN=$(( CLOCK_IN % 100 ))
    OUT_HOUR=$(( CLOCK_OUT / 100 ))
    OUT_MIN=$(( CLOCK_OUT % 100 ))
    
    # Convert to total minutes
    IN_TOTAL_MIN=$(( IN_HOUR * 60 + IN_MIN ))
    OUT_TOTAL_MIN=$(( OUT_HOUR * 60 + OUT_MIN ))
    
    # Calculate worked minutes
    WORKED_MIN=$(( OUT_TOTAL_MIN - IN_TOTAL_MIN ))
    
    # Convert to hours with decimal (multiply by 100 for precision, then divide)
    HOURS=$(( WORKED_MIN * 100 / 60 ))
    
    # Format as decimal hours
    HOURS_INT=$(( HOURS / 100 ))
    HOURS_DEC=$(( HOURS % 100 ))
    
    printf "%s=%.2f\n" "$EMPLOYEE_ID" "$(echo "$HOURS_INT.$HOURS_DEC" | bc -l)" >> "$OUTPUT_FILE"
    
done < "$LOG_FILE"

echo "Calculation complete. Results saved to $OUTPUT_FILE"