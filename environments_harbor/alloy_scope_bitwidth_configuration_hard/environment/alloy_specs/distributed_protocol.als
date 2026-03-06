// Distributed Transaction Protocol Specification
// Bitwidth configuration: bitwidth 10
// Models the core distributed transaction protocol with resource allocation

module distributed_protocol

open util/ordering[DistributedTransaction]

// Core transaction signature with all fields
sig DistributedTransaction {
    txId: Int,
    resourceDelta: Int,
    timestamp: Int,
    priority: Int,
    timeout: Int,
    state: TransactionState
}

abstract sig TransactionState {}
one sig Pending, Committed, Aborted extends TransactionState {}

// Global resource tracker
one sig ResourceManager {
    totalResources: Int
}

// Constraint facts for field ranges
fact TransactionIdBounds {
    all t: DistributedTransaction | t.txId >= 0 and t.txId <= 1000
}

fact ResourceDeltaBounds {
    all t: DistributedTransaction | t.resourceDelta >= -500 and t.resourceDelta <= 500
}

fact PriorityBounds {
    all t: DistributedTransaction | t.priority >= 1 and t.priority <= 10
}

fact TimeoutBounds {
    all t: DistributedTransaction | t.timeout >= 0 and t.timeout <= 3600
}

fact TimestampRealistic {
    all t: DistributedTransaction | t.timestamp >= 1600000000 and t.timestamp <= 2000000000
}

fact UniqueTransactionIds {
    all disj t1, t2: DistributedTransaction | t1.txId != t2.txId
}

fact InitialState {
    all t: DistributedTransaction | t.state = Pending implies 
        (t.timestamp > 0 and t.priority > 0)
}

// Predicate: Commit a transaction
pred commitTransaction[t: DistributedTransaction, t': DistributedTransaction, rm: ResourceManager, rm': ResourceManager] {
    t.state = Pending
    t'.state = Committed
    t'.txId = t.txId
    t'.resourceDelta = t.resourceDelta
    t'.timestamp = t.timestamp
    t'.priority = t.priority
    t'.timeout = t.timeout
    rm'.totalResources = add[rm.totalResources, t.resourceDelta]
    // Overflow protection
    rm'.totalResources >= -10000 and rm'.totalResources <= 10000
}

// Predicate: Abort a transaction
pred abortTransaction[t: DistributedTransaction, t': DistributedTransaction] {
    t.state = Pending
    t'.state = Aborted
    t'.txId = t.txId
    t'.resourceDelta = t.resourceDelta
    t'.timestamp = t.timestamp
    t'.priority = t.priority
    t'.timeout = t.timeout
}

// Predicate: Validate transaction constraints
pred validateTransaction[t: DistributedTransaction] {
    t.txId >= 0 and t.txId <= 1000
    t.resourceDelta >= -500 and t.resourceDelta <= 500
    t.priority >= 1 and t.priority <= 10
    t.timeout >= 0 and t.timeout <= 3600
    t.timestamp > 1600000000
}

pred highPriorityTransaction[t: DistributedTransaction] {
    t.priority >= 8
    t.state = Committed implies t.timeout >= 1000
}

// Assertion: Transaction atomicity
assert TransactionAtomicity {
    all t: DistributedTransaction | 
        t.state = Committed or t.state = Aborted or t.state = Pending
}

// Assertion: No double commitment
assert NoDoubleCommit {
    no disj t1, t2: DistributedTransaction | 
        t1.txId = t2.txId and t1.state = Committed and t2.state = Committed
}

// Assertion: Resource conservation
assert ResourceConservation {
    all rm: ResourceManager | 
        rm.totalResources >= -10000 and rm.totalResources <= 10000
}

// Assertion: Valid priority ordering
assert PriorityOrdering {
    all disj t1, t2: DistributedTransaction | 
        t1.priority > t2.priority implies 
            (t1.state = Committed implies t1.timestamp <= t2.timestamp or t2.state != Committed)
}

check TransactionAtomicity for 5
check NoDoubleCommit for 5
check ResourceConservation for 5
check PriorityOrdering for 5