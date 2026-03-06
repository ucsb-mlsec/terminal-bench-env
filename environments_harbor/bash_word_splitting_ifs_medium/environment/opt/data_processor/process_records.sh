#!/bin/bash

# Data Processing Pipeline for Customer Records
# Generates summary reports from CSV data

INPUT_FILE="/opt/data_processor/sample_data.csv"
OUTPUT_FILE="/opt/data_processor/report.txt"
TEMP_FILE="/tmp/processing_temp.txt"

# Initialize counters
total_balance=0
record_count=0

# Clear previous output
> $OUTPUT_FILE

echo "Processing customer records..." > $TEMP_FILE

# Process CSV file
while IFS=',' read -r name address balance
do
    # Skip header line
    if [ "$name" = "Name" ]; then
        continue
    fi
    
    # Increment counter
    record_count=$((record_count + 1))
    
    # Display customer info (BUGGY: unquoted variables cause word splitting)
    echo "Record $record_count:" >> $OUTPUT_FILE
    echo "  Customer: $name" >> $OUTPUT_FILE
    echo "  Address: $address" >> $OUTPUT_FILE
    echo "  Balance: $balance" >> $OUTPUT_FILE
    
    # Extract numeric balance for calculation (BUGGY: unquoted variable)
    clean_balance=$(echo $balance | tr -d '$' | tr -d ',')
    
    # Add to total (BUGGY: arithmetic with unquoted variable)
    total_balance=$(echo "$total_balance + $clean_balance" | bc)
    
    # Generate temp processing record (BUGGY: unquoted variables)
    echo Processing: $name from $address >> $TEMP_FILE
    
    echo "---" >> $OUTPUT_FILE
done < "$INPUT_FILE"

# Generate summary
echo "" >> $OUTPUT_FILE
echo "SUMMARY REPORT" >> $OUTPUT_FILE
echo "Total Records Processed: $record_count" >> $OUTPUT_FILE
echo "Total Balance: \$$total_balance" >> $OUTPUT_FILE

# Clean temp file
cat $TEMP_FILE >> $OUTPUT_FILE
rm -f $TEMP_FILE

echo "Report generated at $OUTPUT_FILE"