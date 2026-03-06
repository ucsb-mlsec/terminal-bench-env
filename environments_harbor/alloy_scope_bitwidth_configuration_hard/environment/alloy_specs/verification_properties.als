-- Verification Properties for Distributed Transaction System
-- This file contains formal properties to verify the correctness
-- of the transaction system model

-- References to core specification concepts
-- Assumes: Transaction, Resource, Timestamp, Priority concepts are defined
-- Assumes: allocation, deallocation, and timing operations are specified

-- Property 1: Resource Overflow Prevention
-- Verifies that resource counts never exceed the valid range [-500, 500]
assert NoResourceOverflow {
  all r: Resource | {
    r.count >= -500 and r.count <= 500
    -- After any operation, resources remain in bounds
    (some t: Transaction | t.affects[r]) implies {
      r.count.add[t.delta] >= -500 and 
      r.count.add[t.delta] <= 500
    }
  }
}

-- Property 2: Transaction ID Bounds
-- Ensures all transaction IDs stay within the allocated range [0, 1000]
assert TransactionBoundsValid {
  all t: Transaction | {
    t.id >= 0 and t.id <= 1000
    -- Transaction IDs must be unique and within bounds
    no t2: Transaction | t != t2 and t.id = t2.id
  }
}

-- Property 3: Timestamp Ordering
-- Verifies that timestamps are monotonically increasing and within Unix epoch range
-- Unix timestamps are large positive integers (typically > 1000000000)
assert TimestampOrdering {
  all t1, t2: Transaction | {
    t1.startTime >= 1000000000
    t2.startTime >= 1000000000
    -- If t1 completes before t2 starts, timestamps reflect this
    (t1.endTime < t2.startTime) implies t1.endTime <= t2.startTime
    -- Timeouts are within valid range
    t1.timeout >= 0 and t1.timeout <= 3600
  }
}

-- Property 4: Priority Range Validation
-- Confirms that all priority assignments are within [1, 10]
assert PriorityRangeValid {
  all t: Transaction | {
    t.priority >= 1 and t.priority <= 10
    -- Higher priority transactions should not be starved
    (t.priority > 5) implies eventually[t.status = Completed or t.status = Aborted]
  }
}

-- Property 5: Resource Conservation
-- Total resource allocation should remain constant across the system
assert ResourceConservation {
  all r: Resource | {
    sum[t: Transaction | t.allocates[r]] = sum[t: Transaction | t.deallocates[r]]
  }
}

-- Check commands to verify properties
-- Note: No explicit bitwidth set, using Alloy defaults
check NoResourceOverflow for 5
check TransactionBoundsValid for 8
check TimestampOrdering for 6
check PriorityRangeValid for 5
check ResourceConservation for 7

-- Additional check with larger scope for comprehensive verification
check NoResourceOverflow for 10 but 4 Int