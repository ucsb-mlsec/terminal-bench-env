// Helper functions for distributed transaction system verification
// This module provides utility predicates for range checking and validation

module helper_functions

// Generic range checking predicate
pred inRange[value: Int, min: Int, max: Int] {
  value >= min and value <= max
}

// Transaction ID validation (0 to 1000)
pred isValidTransactionId[id: Int] {
  inRange[id, 0, 1000]
}

// Resource amount validation (-500 to 500)
pred isValidResourceAmount[amt: Int] {
  inRange[amt, -500, 500]
}

// Priority validation (1 to 10)
pred isValidPriority[p: Int] {
  inRange[p, 1, 10]
}

// Timeout validation (0 to 3600 seconds)
pred isValidTimeout[t: Int] {
  inRange[t, 0, 3600]
}

// Check if timestamp is a valid Unix epoch value (positive)
pred isValidTimestamp[ts: Int] {
  ts > 0
}

// Compare two timestamps for ordering
pred timestampBefore[ts1: Int, ts2: Int] {
  ts1 < ts2
}

// Calculate if timeout has expired given start timestamp and timeout duration
pred hasTimedOut[startTime: Int, currentTime: Int, timeout: Int] {
  let elapsed = currentTime.minus[startTime] |
    elapsed >= timeout
}

// Check if adding two resource amounts would remain valid
pred canAddResources[amt1: Int, amt2: Int] {
  let sum = amt1.plus[amt2] |
    isValidResourceAmount[sum]
}

// Check if subtracting resource amounts would remain valid
pred canSubtractResources[amt1: Int, amt2: Int] {
  let diff = amt1.minus[amt2] |
    isValidResourceAmount[diff]
}

// Validate that resource transfer maintains bounds
pred validResourceTransfer[from: Int, to: Int, amount: Int] {
  isValidResourceAmount[from] and
  isValidResourceAmount[to] and
  isValidResourceAmount[amount] and
  canSubtractResources[from, amount] and
  canAddResources[to, amount]
}