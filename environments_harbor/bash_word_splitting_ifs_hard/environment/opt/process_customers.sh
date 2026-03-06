#!/bin/bash
# Legacy customer data processing script
# Original author: Unknown
# Purpose: Parse CSV customer records and generate cleaned output

INPUT_FILE="/opt/customer_data.csv"
OUTPUT_FILE="/opt/cleaned_customers.txt"

# Initialize output
echo "Processing customer records..." > "$OUTPUT_FILE"
echo "================================" >> "$OUTPUT_FILE"

# Set IFS to comma for CSV parsing
IFS=','

# Skip header and process each line
line_count=0
while read -r line; do
    line_count=$((line_count + 1))
    
    # Skip header line
    if [ $line_count -eq 1 ]; then
        continue
    fi
    
    # Parse CSV fields - this will split on commas
    read -ra fields <<< "$line"
    
    # Extract fields
    customer_id="${fields[0]}"
    customer_name="${fields[1]}"
    customer_address="${fields[2]}"
    customer_phone="${fields[3]}"
    
    # Process and output customer record
    # This echo will use the modified IFS, causing word splitting
    echo "Customer ID: $customer_id" >> "$OUTPUT_FILE"
    echo "Name: $customer_name" >> "$OUTPUT_FILE"
    echo "Address: $customer_address" >> "$OUTPUT_FILE"
    echo "Phone: $customer_phone" >> "$OUTPUT_FILE"
    echo "---" >> "$OUTPUT_FILE"
    
done < "$INPUT_FILE"

echo "Processing complete. Output saved to $OUTPUT_FILE"