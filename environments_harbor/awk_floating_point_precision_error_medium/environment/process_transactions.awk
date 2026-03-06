BEGIN {
    FS = ","
}

NR > 1 {
    account = $2
    amount = $3
    type = $4
    
    # Buggy floating point arithmetic - no precision handling
    if (type == "CREDIT") {
        balance[account] = balance[account] + amount
    } else if (type == "DEBIT") {
        balance[account] = balance[account] - amount
    }
}

END {
    # Print results without proper formatting
    for (account in balance) {
        accounts[++count] = account
    }
    
    # Bubble sort by account_id
    for (i = 1; i <= count; i++) {
        for (j = i + 1; j <= count; j++) {
            if (accounts[i] > accounts[j]) {
                temp = accounts[i]
                accounts[i] = accounts[j]
                accounts[j] = temp
            }
        }
    }
    
    # Print with potential precision issues
    for (i = 1; i <= count; i++) {
        account = accounts[i]
        # Direct output without proper 2 decimal place formatting
        print account "," balance[account]
    }
}