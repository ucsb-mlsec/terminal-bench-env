// Alloy specification for testing edge cases in distributed transaction system
// Current bitwidth configuration: bitwidth 6 (INADEQUATE - causes overflow)

module edge_cases

// Signature definitions
sig Transaction {
    id: one Int,
    resourceAmount: one Int,
    timestamp: one Int,
    priority: one Int,
    timeout: one Int
}

// Facts enforcing system constraints
fact ValidTransactionIds {
    all t: Transaction | t.id >= 0 and t.id <= 1000
}

fact ValidResourceAmounts {
    all t: Transaction | t.resourceAmount >= -500 and t.resourceAmount <= 500
}

fact ValidPriorities {
    all t: Transaction | t.priority >= 1 and t.priority <= 10
}

fact ValidTimeouts {
    all t: Transaction | t.timeout >= 0 and t.timeout <= 3600
}

// Boundary test predicates
pred maxTransactionId {
    some t: Transaction | t.id = 1000
}

pred minResourceAmount {
    some t: Transaction | t.resourceAmount = -500
}

pred maxResourceAmount {
    some t: Transaction | t.resourceAmount = 500
}

pred maxPriority {
    some t: Transaction | t.priority = 10
}

pred maxTimeout {
    some t: Transaction | t.timeout = 3600
}

pred largeTimestamp {
    some t: Transaction | t.timestamp = 2000000000
}

pred minBoundaryValues {
    some t: Transaction | t.id = 0 and t.resourceAmount = -500 and t.priority = 1 and t.timeout = 0
}

// Run commands for edge case verification
run maxTransactionId for 3
run minResourceAmount for 3
run maxResourceAmount for 3
run maxPriority for 3
run maxTimeout for 3
run largeTimestamp for 3
run minBoundaryValues for 3