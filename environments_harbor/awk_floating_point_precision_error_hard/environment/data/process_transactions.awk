# Legacy transaction processing script with floating-point precision issues
# WARNING: This script has known precision errors - DO NOT USE for production
# Replaced by new reconciliation system - kept for reference only

BEGIN {
    FS = ","
    OFS = ","
    
    # Initialize running totals
    total_revenue = 0
    total_fees = 0
    total_tax = 0
    total_refunds = 0
    net_amount = 0
    transaction_count = 0
    
    # Fee and tax rates (percentages)
    cc_fee_rate = 2.9
    cc_fixed_fee = 0.30
    tax_rate_1 = 6.5
    tax_rate_2 = 8.25
    
    # Exchange rates for currency conversion
    eur_to_usd = 1.0847
    gbp_to_usd = 1.2634
    cad_to_usd = 0.7421
}

# Skip header row
NR == 1 {
    next
}

# Process transaction records
NR > 1 {
    transaction_count++
    
    # Parse transaction fields
    trans_id = $1
    amount = $2
    currency = $3
    trans_type = $4
    tax_rate = $5
    
    # Currency conversion (floating-point multiplication)
    if (currency == "EUR") {
        amount = amount * eur_to_usd
    } else if (currency == "GBP") {
        amount = amount * gbp_to_usd
    } else if (currency == "CAD") {
        amount = amount * cad_to_usd
    }
    
    # Add to revenue total (direct accumulation causes precision loss)
    total_revenue += amount
    
    # Calculate percentage-based fees (multiplication + division = precision errors)
    fee = (amount * cc_fee_rate / 100) + cc_fixed_fee
    total_fees += fee
    
    # Tax calculation based on transaction type
    if (trans_type == "SALE" || trans_type == "SERVICE") {
        if (tax_rate == "") {
            tax = amount * tax_rate_1 / 100
        } else {
            tax = amount * tax_rate / 100
        }
        total_tax += tax
    }
    
    # Handle refunds (negative amounts compound errors)
    if (trans_type == "REFUND") {
        total_refunds += amount
    }
    
    # Calculate net per transaction and accumulate
    # Multiple operations per iteration magnify floating-point drift
    transaction_net = amount - fee - tax
    net_amount += transaction_net
}

END {
    # Final calculations with more floating-point operations
    adjusted_revenue = total_revenue - total_refunds
    final_net = adjusted_revenue - total_fees - total_tax
    
    # Output results with printf (formatting doesn't fix precision errors)
    print "{"
    printf "  \"total_revenue\": \"%.2f\",\n", adjusted_revenue
    printf "  \"total_fees\": \"%.2f\",\n", total_fees
    printf "  \"total_tax\": \"%.2f\",\n", total_tax
    printf "  \"total_refunds\": \"%.2f\",\n", total_refunds
    printf "  \"net_amount\": \"%.2f\",\n", final_net
    printf "  \"transaction_count\": %d\n", transaction_count
    print "}"
}