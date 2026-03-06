#!/bin/bash

# Critical Data Reconciliation Pipeline
# This script compares regional transaction files against the master file
# to identify discrepancies in financial data.

# Master file location
MASTER_FILE="/data/master/expected_transactions.txt"

# Array of regional offices
REGIONS=("us-east" "us-west" "eu-central" "asia-pacific" "latam")

# Initialize total difference counter
TOTAL_DIFFERENCES=0

echo "Starting reconciliation process..."
echo "Master file: $MASTER_FILE"
echo "================================"

# Loop through each region and perform comparison
for region in "${REGIONS[@]}"; do
    REGIONAL_FILE="/data/regions/${region}/transactions.txt"
    
    echo "Processing region: $region"
    
    # Check if regional file exists
    if [[ ! -f "$REGIONAL_FILE" ]]; then
        echo "ERROR: Regional file not found: $REGIONAL_FILE"
        continue
    fi
    
    # Use comm to compare files
    # comm -3 suppresses lines that appear in both files
    # This shows only the differences (lines unique to either file)
    DIFFERENCES=$(comm -3 "$MASTER_FILE" "$REGIONAL_FILE" | wc -l)
    
    echo "Region $region: $DIFFERENCES differences found"
    
    # Add to total
    TOTAL_DIFFERENCES=$((TOTAL_DIFFERENCES + DIFFERENCES))
    
    echo "---"
done

echo "================================"
echo "Total differences across all regions: $TOTAL_DIFFERENCES"
echo "Reconciliation process complete."

# Exit with error code if differences found
if [[ $TOTAL_DIFFERENCES -gt 0 ]]; then
    echo "WARNING: Discrepancies detected in financial data!"
    exit 1
else
    echo "SUCCESS: All regional data matches master file."
    exit 0
fi
```

```bash
#!/bin/bash

# Solution script to diagnose and fix locale-dependent sorting issues
# in the data reconciliation pipeline

# Create solution directory
mkdir -p /solution

# Output file
REPORT_FILE="/solution/fix_report.txt"

# Master file location
MASTER_FILE="/data/master/expected_transactions.txt"

# Array of regional offices
REGIONS=("us-east" "us-west" "eu-central" "asia-pacific" "latam")

# Common locales to test
LOCALES=("C" "en_US.UTF-8" "en_GB.UTF-8" "de_DE.UTF-8" "ja_JP.UTF-8" "es_ES.UTF-8")

echo "Diagnosing locale-dependent sorting issues..."

# Step 1: Determine the locale used for the master file
# We'll test which locale produces a sort order that matches the master file
MASTER_LOCALE=""

for test_locale in "${LOCALES[@]}"; do
    # Sort master file with test locale and compare
    LC_COLLATE=$test_locale sort "$MASTER_FILE" > /tmp/master_test_sorted.txt
    
    if diff -q "$MASTER_FILE" /tmp/master_test_sorted.txt > /dev/null 2>&1; then
        MASTER_LOCALE="$test_locale"
        echo "Master file locale identified: $MASTER_LOCALE"
        break
    fi
done

# If no locale matches exactly, check if file is already sorted with C locale (most common)
if [[ -z "$MASTER_LOCALE" ]]; then
    # Default to C locale as it's most predictable
    MASTER_LOCALE="C"
    echo "Using default locale: $MASTER_LOCALE"
fi

# Step 2: Calculate total mismatches BEFORE fix
echo "Calculating mismatches before fix..."
TOTAL_MISMATCHES_BEFORE=0

for region in "${REGIONS[@]}"; do
    REGIONAL_FILE="/data/regions/${region}/transactions.txt"
    
    if [[ -f "$REGIONAL_FILE" ]]; then
        DIFFERENCES=$(comm -3 "$MASTER_FILE" "$REGIONAL_FILE" 2>/dev/null | wc -l)
        TOTAL_MISMATCHES_BEFORE=$((TOTAL_MISMATCHES_BEFORE + DIFFERENCES))
    fi
done

echo "Total mismatches before fix: $TOTAL_MISMATCHES_BEFORE"

# Step 3: Identify which regional files need re-sorting
echo "Identifying files with sorting issues..."
REGIONS_NEEDING_FIX=()

for region in "${REGIONS[@]}"; do
    REGIONAL_FILE="/data/regions/${region}/transactions.txt"
    
    if [[ -f "$REGIONAL_FILE" ]]; then
        # Check if the file is sorted according to master locale
        LC_COLLATE=$MASTER_LOCALE sort -c "$REGIONAL_FILE" 2>/dev/null
        
        if [[ $? -ne 0 ]]; then
            echo "Region $region needs re-sorting"
            REGIONS_NEEDING_FIX+=("$region")
        else
            # Even if sorted, check if it matches master's sort order by testing comm
            DIFFERENCES=$(comm -3 "$MASTER_FILE" "$REGIONAL_FILE" 2>/dev/null | wc -l)
            
            # If there are differences, re-sort to ensure consistency
            if [[ $DIFFERENCES -gt 0 ]]; then
                # Check if re-sorting would help
                LC_COLLATE=$MASTER_LOCALE sort "$REGIONAL_FILE" > /tmp/test_resort_${region}.txt
                NEW_DIFFERENCES=$(comm -3 "$MASTER_FILE" /tmp/test_resort_${region}.txt 2>/dev/null | wc -l)
                
                if [[ $NEW_DIFFERENCES -lt $DIFFERENCES ]]; then
                    echo "Region $region needs re-sorting (has $DIFFERENCES differences)"
                    REGIONS_NEEDING_FIX+=("$region")
                fi
                rm -f /tmp/test_resort_${region}.txt
            fi
        fi
    fi
done

# Step 4: Fix all affected regional files
echo "Fixing regional files..."

for region in "${REGIONS_NEEDING_FIX[@]}"; do
    REGIONAL_FILE="/data/regions/${region}/transactions.txt"
    BACKUP_FILE="/data/regions/${region}/transactions.txt.backup"
    
    # Create backup
    cp "$REGIONAL_FILE" "$BACKUP_FILE"
    
    # Re-sort with master locale
    LC_COLLATE=$MASTER_LOCALE sort "$BACKUP_FILE" > "$REGIONAL_FILE"
    
    echo "Fixed: $region"
done

# Step 5: Verify and calculate mismatches AFTER fix
echo "Verifying fixes..."
TOTAL_MISMATCHES_AFTER=0

for region in "${REGIONS[@]}"; do
    REGIONAL_FILE="/data/regions/${region}/transactions.txt"
    
    if [[ -f "$REGIONAL_FILE" ]]; then
        DIFFERENCES=$(comm -3 "$MASTER_FILE" "$REGIONAL_FILE" 2>/dev/null | wc -l)
        TOTAL_MISMATCHES_AFTER=$((TOTAL_MISMATCHES_AFTER + DIFFERENCES))
        
        if [[ $DIFFERENCES -gt 0 ]]; then
            echo "Region $region still has $DIFFERENCES actual differences"
        fi
    fi
done

echo "Total mismatches after fix: $TOTAL_MISMATCHES_AFTER"

# Step 6: Generate report
REGIONS_FIXED_STR=$(IFS=,; echo "${REGIONS_NEEDING_FIX[*]}")

cat > "$REPORT_FILE" << EOF
MASTER_LOCALE=$MASTER_LOCALE
REGIONS_FIXED=$REGIONS_FIXED_STR
TOTAL_MISMATCHES_BEFORE=$TOTAL_MISMATCHES_BEFORE
TOTAL_MISMATCHES_AFTER=$TOTAL_MISMATCHES_AFTER
EOF

echo "Report generated: $REPORT_FILE"
cat "$REPORT_FILE"

# Verify reconciliation now works
echo ""
echo "Running final reconciliation check..."
/opt/pipeline/reconcile.sh

echo ""
echo "Solution complete!"