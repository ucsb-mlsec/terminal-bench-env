(set-logic QF_AUFLIA)

; Array variable declarations
(declare-fun arr1 () (Array Int Int))
(declare-fun arr2 () (Array Int Int))
(declare-fun arr3 () (Array Int Int))

; Assertion 1: Deeply nested store/select operations (nesting depth 8)
(assert (= (select (store (select (store (select (store (select (store arr1 0 10) 1) 2 20) 3) 4 30) 5) 6 40) 7) 40))

; Assertion 2: Another deeply nested expression (nesting depth 7)
(assert (> (select (store (select (store (select (store (select arr2 0) 1 5) 2) 3 15) 4) 5 25) 6) 20))

; Assertion 3: Nested operations with multiple arrays (nesting depth 7)
(assert (= (select (store (select (store (select (store arr3 10 100) 11) 12 200) 13) 14 300) 15) 300))

; Assertion 4: Mixed nested operations (nesting depth 6)
(assert (< (select (store (select (store (select arr1 20) 21 50) 22) 23 60) 24) 100))

; Assertion 5: Simple array operations for variety
(assert (= (select (store arr2 50 500) 50) 500))

; Assertion 6: Another simple operation
(assert (= (select arr3 100) (select (store arr3 100 1000) 100)))

(check-sat)