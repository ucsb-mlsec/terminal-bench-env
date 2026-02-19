#!/bin/bash

# Payroll calculation script
# Reads timecards and calculates daily pay for each employee

INPUT_FILE="/opt/payroll/data/timecards.csv"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found"
    exit 1
fi

# Print header for output
echo "EMPLOYEE_ID,DATE,HOURS_WORKED,DAILY_PAY"

# Read CSV file line by line, skip header
tail -n +2 "$INPUT_FILE" | while IFS=, read -r EMPLOYEE_ID DATE CLOCK_IN CLOCK_OUT HOURLY_RATE; do
    
    # Extract hours and minutes from clock-in time (HH:MM format)
    hour_in=$(echo "$CLOCK_IN" | cut -d':' -f1)
    minute_in=$(echo "$CLOCK_IN" | cut -d':' -f2)
    
    # Extract hours and minutes from clock-out time (HH:MM format)
    hour_out=$(echo "$CLOCK_OUT" | cut -d':' -f1)
    minute_out=$(echo "$CLOCK_OUT" | cut -d':' -f2)
    
    # Convert clock-in time to total minutes since midnight
    # Note: Direct arithmetic on hour values with leading zeros
    total_minutes_in=$(( hour_in * 60 + minute_in ))
    
    # Convert clock-out time to total minutes since midnight
    total_minutes_out=$(( hour_out * 60 + minute_out ))
    
    # Calculate total minutes worked
    minutes_worked=$(( total_minutes_out - total_minutes_in ))
    
    # Convert minutes to decimal hours (divide by 60)
    # Using bc for decimal division
    hours_worked=$(echo "scale=2; $minutes_worked / 60" | bc)
    
    # Calculate daily pay (hours * hourly rate)
    daily_pay=$(echo "scale=2; $hours_worked * $HOURLY_RATE" | bc)
    
    # Output the results
    echo "$EMPLOYEE_ID,$DATE,$hours_worked,$daily_pay"
    
done