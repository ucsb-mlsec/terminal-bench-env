# Daily Transaction Aggregator
# Aggregates transactions by date and calculates daily totals
# WARNING: This script demonstrates floating-point precision errors in AWK

BEGIN {
    FS = ","
    # Tax rate for calculations
    TAX_RATE = 0.0825
    # Fee percentage
    FEE_PERCENT = 0.029
    FEE_FIXED = 0.30
    
    print "Starting daily aggregation..."
}

# Skip header line
NR == 1 {
    next
}

# Process each transaction record
# Expected format: date,transaction_id,amount,currency,type
{
    date = $1
    amount = $3
    type = $5
    
    # Accumulate daily totals - this causes precision errors
    daily_total[date] += amount
    daily_count[date]++
    
    # Calculate fees with floating-point arithmetic
    fee = (amount * FEE_PERCENT) + FEE_FIXED
    daily_fees[date] += fee
    
    # Calculate tax on the amount
    tax = amount * TAX_RATE
    daily_tax[date] += tax
    
    # Track cumulative running total across all dates
    cumulative_total += amount
    
    # Calculate net amount (amount - fees - tax)
    net = amount - fee - tax
    daily_net[date] += net
    
    # Track overall statistics
    total_transactions++
    grand_total += amount
}

END {
    print "\n=== Daily Transaction Summary ==="
    print "Date,Total,Fees,Tax,Net,Count,Average"
    
    # Sort dates and print summary
    n = asorti(daily_total, sorted_dates)
    
    running_total = 0
    for (i = 1; i <= n; i++) {
        date = sorted_dates[i]
        total = daily_total[date]
        fees = daily_fees[date]
        tax = daily_tax[date]
        net = daily_net[date]
        count = daily_count[date]
        
        # Calculate average - more division means more precision loss
        average = total / count
        
        # Update running total - accumulates errors across days
        running_total += total
        
        # Print with 2 decimal places - masks underlying precision errors
        printf "%s,%.2f,%.2f,%.2f,%.2f,%d,%.2f\n", 
            date, total, fees, tax, net, count, average
    }
    
    # Calculate grand averages - compounds all previous errors
    grand_average = grand_total / total_transactions
    
    print "\n=== Overall Totals ==="
    printf "Grand Total: %.2f\n", grand_total
    printf "Cumulative Total: %.2f\n", cumulative_total
    printf "Running Total: %.2f\n", running_total
    printf "Total Transactions: %d\n", total_transactions
    printf "Average Transaction: %.2f\n", grand_average
    
    # These three totals should be identical but may differ due to precision errors
    if (grand_total != cumulative_total || grand_total != running_total) {
        print "WARNING: Totals do not match - precision errors detected!"
        printf "Difference: %.10f\n", (grand_total - cumulative_total)
    }
}