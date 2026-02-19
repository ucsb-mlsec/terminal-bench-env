BEGIN {
    FS = ","
    print "Processing Regional Sales Data"
    print "================================"
}

NR == 1 {
    print "Starting file: " FILENAME
    total_amount = 0
}

NR > 1 {
    region = $1
    product = $2
    amount = $3
    
    print "Line " NR ": Processing " region " - " product " - $" amount
    
    total_amount += amount
    line_count++
}

NR > 1 && NR <= 10 {
    valid_data++
}

FNR == 1 {
    if (prev_file != "" && prev_file != FILENAME) {
        print "\nRegional Summary for " prev_file ":"
        print "Total Sales: $" total_amount
        print "Lines Processed: " line_count
        print "================================\n"
    }
    prev_file = FILENAME
}

END {
    print "\nRegional Summary for " FILENAME ":"
    print "Total Sales: $" total_amount
    print "Lines Processed: " line_count
    print "================================"
    print "\nOverall Total Lines: " NR
    print "Valid Data Entries: " valid_data
}