#!/bin/bash

# Stage 4: Final Counting Stage
# This stage reads the deduplicated IP addresses from stage 3
# and counts the total number of unique visitors

# Input file: IP addresses that should already be unique from stage 3
INPUT_FILE="/opt/analytics/data/output/stage3_unique.txt"

# Output file: Final count of unique visitors
OUTPUT_FILE="/opt/analytics/data/output/final_count.txt"

# Count the number of lines in the input file
# Each line represents one unique visitor IP address
# Use 'wc -l' to count lines and extract only the number
COUNT=$(wc -l < "$INPUT_FILE")

# Output just the number to the final count file
echo "$COUNT" > "$OUTPUT_FILE"

# Stage 4 complete - unique visitor count has been written to final_count.txt