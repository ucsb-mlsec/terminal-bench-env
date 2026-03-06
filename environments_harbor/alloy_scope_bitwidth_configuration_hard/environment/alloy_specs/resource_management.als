module resource_management

// bitwidth 8

sig Resource {
    amount: Int,
    resourceId: Int
}

sig ResourceHolder {
    resources: set Resource
}

// Constraint: Resource amounts must be between -500 and 500
fact ValidResourceAmounts {
    all r: Resource | r.amount >= -500 and r.amount <= 500
}

// Constraint: Resource IDs must be positive
fact ValidResourceIds {
    all r: Resource | r.resourceId >= 0
}

// Constraint: Each resource has unique ID
fact UniqueResourceIds {
    all disj r1, r2: Resource | r1.resourceId != r2.resourceId
}

// Constraint: Total resources must balance to zero (conservation)
fact ResourceConservation {
    sum r: Resource | r.amount = 0
}

// Predicate: Allocate a new resource with specified amount
pred allocateResource[holder: ResourceHolder, r: Resource, amt: Int] {
    amt >= -500 and amt <= 500
    r.amount = amt
    r not in holder.resources
    holder.resources' = holder.resources + r
}

// Predicate: Deallocate an existing resource
pred deallocateResource[holder: ResourceHolder, r: Resource] {
    r in holder.resources
    holder.resources' = holder.resources - r
}

// Predicate: Transfer resources between holders
pred transferResource[from: ResourceHolder, to: ResourceHolder, r: Resource, amt: Int] {
    r in from.resources
    amt >= -500 and amt <= 500
    amt <= r.amount
    r.amount' = r.amount - amt
    // Create corresponding resource in destination
    some r2: Resource | {
        r2 not in to.resources
        r2.amount' = amt
        to.resources' = to.resources + r2
    }
    from.resources' = from.resources
}

// Predicate: Check if resource holder has sufficient resources
pred hasSufficientResources[holder: ResourceHolder, requiredAmount: Int] {
    sum r: holder.resources | r.amount >= requiredAmount
}

// Assertion: Total resources always balance to zero
assert ResourceBalancePreserved {
    always (sum r: Resource | r.amount = 0)
}

// Assertion: No resource exceeds valid bounds
assert NoOverflow {
    always (all r: Resource | r.amount >= -500 and r.amount <= 500)
}

// Assertion: Resource amounts are within safe integer bounds
assert SafeIntegerBounds {
    all r: Resource | {
        r.amount >= -500 and r.amount <= 500
        r.resourceId >= 0 and r.resourceId <= 1000
    }
}

// Check assertions with appropriate scope
check ResourceBalancePreserved for 5
check NoOverflow for 5
check SafeIntegerBounds for 5