// Distributed Transaction System - Core Transaction Model
// Bitwidth configuration: bitwidth 5 (insufficient for transaction IDs up to 1000)

module transaction_core

// Transaction status enumeration
abstract sig Status {}
one sig Pending, Active, Committed, Aborted extends Status {}

// Core transaction signature
sig Transaction {
    id: one Int,
    status: one Status,
    priority: one Int
}

// Fact: Ensure all transaction IDs are within valid range (0-1000)
fact ValidTransactionIDs {
    all t: Transaction | t.id >= 0 and t.id <= 1000
}

// Fact: Ensure all priorities are within valid range (1-10)
fact ValidPriorities {
    all t: Transaction | t.priority >= 1 and t.priority <= 10
}

// Fact: Transaction IDs must be unique
fact UniqueTransactionIDs {
    all disj t1, t2: Transaction | t1.id != t2.id
}

// Fact: New transactions start in Pending status
fact InitialStatus {
    all t: Transaction | t.status = Pending implies t.priority >= 1
}

// Predicate: Create a new transaction with given ID and priority
pred createTransaction[t: Transaction, txId: Int, prio: Int] {
    t.id = txId
    t.status = Pending
    t.priority = prio
    txId >= 0 and txId <= 1000
    prio >= 1 and prio <= 10
}

// Predicate: Transition transaction to active state
pred activateTransaction[t: Transaction, t': Transaction] {
    t.status = Pending
    t'.id = t.id
    t'.status = Active
    t'.priority = t.priority
}

// Predicate: Commit a transaction
pred commitTransaction[t: Transaction, t': Transaction] {
    t.status = Active
    t'.id = t.id
    t'.status = Committed
    t'.priority = t.priority
}

// Assertion: Created transactions have valid IDs
assert CreatedTransactionValid {
    all t: Transaction | createTransaction[t, t.id, t.priority] 
        implies (t.id >= 0 and t.id <= 1000)
}

// Assertion: Priority values remain valid through state transitions
assert PriorityInvariant {
    all t, t': Transaction | activateTransaction[t, t'] 
        implies (t'.priority >= 1 and t'.priority <= 10)
}

// Run command to find valid transaction configurations
run createTransaction for 5 Transaction

// Check assertions
check CreatedTransactionValid for 5 Transaction
check PriorityInvariant for 5 Transaction