(set-logic QF_ABV)

; Declare array variables
(declare-fun arr1 () (Array (_ BitVec 32) (_ BitVec 32)))
(declare-fun arr2 () (Array (_ BitVec 32) (_ BitVec 32)))
(declare-fun arr3 () (Array (_ BitVec 32) (_ BitVec 32)))
(declare-fun arr4 () (Array (_ BitVec 32) (_ BitVec 32)))
(declare-fun arr5 () (Array (_ BitVec 32) (_ BitVec 32)))
(declare-fun arr6 () (Array (_ BitVec 32) (_ BitVec 32)))
(declare-fun arr7 () (Array (_ BitVec 32) (_ BitVec 32)))
(declare-fun arr8 () (Array (_ BitVec 32) (_ BitVec 32)))

; Declare index and value variables
(declare-fun idx1 () (_ BitVec 32))
(declare-fun idx2 () (_ BitVec 32))
(declare-fun idx3 () (_ BitVec 32))
(declare-fun val1 () (_ BitVec 32))
(declare-fun val2 () (_ BitVec 32))

; Assertions with array operations - building up complexity
(assert (= (select arr1 idx1) val1))
(assert (= (select arr2 idx2) val2))
(assert (= (select arr3 idx3) (select arr1 idx1)))
(assert (= (select (store arr1 idx1 val1) idx2) (select arr2 idx2)))
(assert (= (select (store arr2 idx2 val2) idx3) (select arr3 idx3)))

; More complex nested operations
(assert (= (select (store (store arr1 idx1 val1) idx2 val2) idx3) (select arr4 idx1)))
(assert (= (select (store (store arr2 idx2 val2) idx3 val1) idx1) (select arr5 idx2)))
(assert (= (select (store (store arr3 idx3 val1) idx1 val2) idx2) (select arr6 idx3)))

; Chains of select operations
(assert (= (select arr1 (select arr2 idx1)) (select arr3 idx2)))
(assert (= (select arr2 (select arr3 idx2)) (select arr4 idx3)))
(assert (= (select arr3 (select arr4 idx3)) (select arr5 idx1)))
(assert (= (select arr4 (select arr5 idx1)) (select arr6 idx2)))
(assert (= (select arr5 (select arr6 idx2)) (select arr7 idx3)))
(assert (= (select arr6 (select arr7 idx3)) (select arr8 idx1)))

; Multiple store operations
(assert (= (select (store arr1 idx1 (select arr2 idx2)) idx3) val1))
(assert (= (select (store arr2 idx2 (select arr3 idx3)) idx1) val2))
(assert (= (select (store arr3 idx3 (select arr4 idx1)) idx2) val1))
(assert (= (select (store arr4 idx1 (select arr5 idx2)) idx3) val2))
(assert (= (select (store arr5 idx2 (select arr6 idx3)) idx1) val1))

; Deeper nesting with store and select combinations
(assert (= (select (store (store arr1 idx1 val1) idx2 (select arr2 idx3)) idx1) val2))
(assert (= (select (store (store arr2 idx2 val2) idx3 (select arr3 idx1)) idx2) val1))
(assert (= (select (store (store arr3 idx3 val1) idx1 (select arr4 idx2)) idx3) val2))
(assert (= (select (store (store arr4 idx1 val2) idx2 (select arr5 idx3)) idx1) val1))

; More operations to increase count
(assert (= (select arr7 idx1) (select arr8 idx2)))
(assert (= (select arr8 idx2) (select arr1 idx3)))
(assert (= (select (store arr7 idx1 val1) idx2) (select arr2 idx1)))
(assert (= (select (store arr8 idx2 val2) idx3) (select arr3 idx2)))

; Additional complex patterns
(assert (= (select (store arr1 idx1 (select (store arr2 idx2 val1) idx3)) idx1) val2))
(assert (= (select (store arr2 idx2 (select (store arr3 idx3 val2) idx1)) idx2) val1))
(assert (= (select (store arr3 idx3 (select (store arr4 idx1 val1) idx2)) idx3) val2))
(assert (= (select (store arr4 idx1 (select (store arr5 idx2 val2) idx3)) idx1) val1))
(assert (= (select (store arr5 idx2 (select (store arr6 idx3 val1) idx1)) idx2) val2))

; More operations with different combinations
(assert (= (select arr1 idx2) (select (store arr2 idx3 val1) idx1)))
(assert (= (select arr2 idx3) (select (store arr3 idx1 val2) idx2)))
(assert (= (select arr3 idx1) (select (store arr4 idx2 val1) idx3)))
(assert (= (select arr4 idx2) (select (store arr5 idx3 val2) idx1)))
(assert (= (select arr5 idx3) (select (store arr6 idx1 val1) idx2)))
(assert (= (select arr6 idx1) (select (store arr7 idx2 val2) idx3)))
(assert (= (select arr7 idx2) (select (store arr8 idx3 val1) idx1)))
(assert (= (select arr8 idx3) (select (store arr1 idx1 val2) idx2)))

; Additional store chains
(assert (= (store (store arr1 idx1 val1) idx2 val2) (store (store arr2 idx2 val2) idx3 val1)))
(assert (= (store (store arr3 idx3 val1) idx1 val2) (store (store arr4 idx1 val2) idx2 val1)))
(assert (= (store (store arr5 idx2 val2) idx3 val1) (store (store arr6 idx3 val1) idx1 val2)))

; More select operations to reach target count
(assert (= (select arr1 idx1) (select arr2 idx1)))
(assert (= (select arr2 idx2) (select arr3 idx2)))
(assert (= (select arr3 idx3) (select arr4 idx3)))
(assert (= (select arr4 idx1) (select arr5 idx1)))
(assert (= (select arr5 idx2) (select arr6 idx2)))
(assert (= (select arr6 idx3) (select arr7 idx3)))
(assert (= (select arr7 idx1) (select arr8 idx1)))

; Additional complex nested operations
(assert (= (select (store (store (store arr1 idx1 val1) idx2 val2) idx3 val1) idx1) val2))
(assert (= (select (store (store (store arr2 idx2 val2) idx3 val1) idx1 val2) idx2) val1))
(assert (= (select (store (store (store arr3 idx3 val1) idx1 val2) idx2 val1) idx3) val2))

(check-sat)