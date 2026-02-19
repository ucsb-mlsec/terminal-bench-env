(set-logic QF_ABV)

; Array variable declarations
(declare-fun arr1 () (Array (_ BitVec 32) (_ BitVec 32)))
(declare-fun arr2 () (Array (_ BitVec 32) (_ BitVec 32)))
(declare-fun arr3 () (Array (_ BitVec 32) (_ BitVec 32)))

; Index and value declarations
(declare-fun idx1 () (_ BitVec 32))
(declare-fun idx2 () (_ BitVec 32))
(declare-fun val1 () (_ BitVec 32))
(declare-fun val2 () (_ BitVec 32))

; Assertion 1: Simple store and select on arr1
(assert (= (select (store arr1 idx1 val1) idx1) val1))

; Assertion 2: Basic array equality check
(assert (= (select arr1 idx2) (select arr2 idx2)))

; Assertion 3: Store operation with nested select (depth 2)
(assert (= (select (store arr2 idx1 (select arr1 idx2)) idx1) (select arr1 idx2)))

; Assertion 4: Multiple operations on arr3
(assert (not (= (select arr3 idx1) val2)))

; Assertion 5: Combined store operations
(assert (= (select (store arr3 idx2 val2) idx2) val2))

(check-sat)