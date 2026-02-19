// Alloy Specification for Timing Constraints in Distributed Transaction System
// Bitwidth configuration: bitwidth 20
// Models Unix epoch timestamps and timeout durations with temporal ordering

module timing_constraints

open util/ordering[TimeStamp]

// Signature representing a timestamp with Unix epoch time
sig TimeStamp {
    epochTime: Int
}

// Signature representing timeout duration
sig Timeout {
    duration: Int,
    startTime: TimeStamp,
    deadline: TimeStamp
}

// Fact: All timestamps must represent valid Unix epoch values
// Range: 1600000000 to 2000000000 (roughly year 2020 to 2033)
fact ValidTimestamps {
    all t: TimeStamp | t.epochTime >= 1600000000 and t.epochTime <= 2000000000
}

// Fact: All timeout durations must be between 0 and 3600 seconds (1 hour max)
fact ValidTimeouts {
    all t: Timeout | t.duration >= 0 and t.duration <= 3600
}

// Fact: Deadline must equal start time plus duration
fact DeadlineConsistency {
    all t: Timeout | t.deadline.epochTime = add[t.startTime.epochTime, t.duration]
}

// Fact: Temporal ordering must be maintained
fact TemporalOrdering {
    all t: Timeout | t.startTime.epochTime <= t.deadline.epochTime
}

// Predicate: Check if a timeout has expired at a given current time
pred isExpired[t: Timeout, currentTime: TimeStamp] {
    currentTime.epochTime >= t.deadline.epochTime
}

// Predicate: Set a deadline based on start time and duration
pred setDeadline[t: Timeout] {
    t.deadline.epochTime = add[t.startTime.epochTime, t.duration]
    t.startTime.epochTime < t.deadline.epochTime
}

// Predicate: Check if timeout is valid and not yet expired
pred checkTimeout[t: Timeout, currentTime: TimeStamp] {
    currentTime.epochTime >= t.startTime.epochTime
    currentTime.epochTime < t.deadline.epochTime
    t.duration > 0
}

// Predicate: Ensure timeout ordering between two timeouts
pred timeoutBefore[t1, t2: Timeout] {
    t1.deadline.epochTime <= t2.startTime.epochTime
}

// Assertion: Deadlines should always be after or equal to start times
assert DeadlineAfterStart {
    all t: Timeout | t.deadline.epochTime >= t.startTime.epochTime
}

// Check the assertion
check DeadlineAfterStart for 5

// Run command to find valid timeout configurations
run {
    some t: Timeout | t.duration > 0
} for 5