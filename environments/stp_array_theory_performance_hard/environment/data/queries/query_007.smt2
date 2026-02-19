(set-logic QF_ALIA)

; Declare 14 array variables to exceed the 10 distinct array threshold
(declare-fun arr1 () (Array Int Int))
(declare-fun arr2 () (Array Int Int))
(declare-fun arr3 () (Array Int Int))
(declare-fun arr4 () (Array Int Int))
(declare-fun arr5 () (Array Int Int))
(declare-fun arr6 () (Array Int Int))
(declare-fun arr7 () (Array Int Int))
(declare-fun arr8 () (Array Int Int))
(declare-fun arr9 () (Array Int Int))
(declare-fun arr10 () (Array Int Int))
(declare-fun arr11 () (Array Int Int))
(declare-fun arr12 () (Array Int Int))
(declare-fun arr13 () (Array Int Int))
(declare-fun arr14 () (Array Int Int))

; Declare index and value variables
(declare-fun i1 () Int)
(declare-fun i2 () Int)
(declare-fun i3 () Int)
(declare-fun i4 () Int)
(declare-fun i5 () Int)
(declare-fun v1 () Int)
(declare-fun v2 () Int)
(declare-fun v3 () Int)

; Complex nested expression with depth 7
(assert (= (select (store (select (store (select (store (select arr1 i1) i2) (select arr2 i3) v1) i4) i5) (select arr3 (select arr4 i1))) v2) i1) v3))

; Another deeply nested expression with depth 6
(assert (> (select (store (select (store (select (store arr5 i2) i3) v1) i4) (select arr6 i5)) i1) 0))

; Multiple assertions with various array operations
(assert (= (select arr1 i1) (select arr2 i2)))
(assert (= (select arr3 i3) (select arr4 i4)))
(assert (= (select arr5 i5) (select arr6 i1)))
(assert (= (select arr7 i2) (select arr8 i3)))
(assert (= (select arr9 i4) (select arr10 i5)))
(assert (= (select arr11 i1) (select arr12 i2)))
(assert (= (select arr13 i3) (select arr14 i4)))

; Complex store operations
(assert (= (store (store (store arr1 i1 v1) i2 v2) i3 v3) arr2))
(assert (= (store (store arr3 i4 v1) i5 v2) arr4))
(assert (= (store arr5 i1 (select arr6 i2)) arr7))
(assert (= (store arr8 i3 (select arr9 i4)) arr10))
(assert (= (store arr11 i5 (select arr12 i1)) arr13))

; Nested select operations
(assert (= (select (select (store arr1 i1 v1) i2) i3) v2))
(assert (< (select arr2 (select arr3 i1)) (select arr4 (select arr5 i2))))
(assert (> (select arr6 (select arr7 i3)) (select arr8 (select arr9 i4))))

; More complex nested store-select combinations
(assert (= (select (store (store arr10 i1 v1) i2 (select arr11 i3)) i4) v2))
(assert (= (select (store (store arr12 i5 v2) i1 (select arr13 i2)) i3) v3))
(assert (= (select (store arr14 i4 (select (store arr1 i5 v1) i1)) i2) v2))

; Additional operations to reach 55-65 operations
(assert (= (store arr2 i1 (select arr3 i2)) (store arr4 i3 (select arr5 i4))))
(assert (= (store arr6 i5 (select arr7 i1)) (store arr8 i2 (select arr9 i3))))
(assert (= (store arr10 i4 (select arr11 i5)) (store arr12 i1 (select arr13 i2))))
(assert (= (select (store arr14 i3 v1) i4) (select (store arr1 i5 v2) i1)))
(assert (= (select (store arr2 i2 v3) i3) (select (store arr3 i4 v1) i5)))
(assert (= (select (store arr4 i1 v2) i2) (select (store arr5 i3 v3) i4)))
(assert (= (select (store arr6 i5 v1) i1) (select (store arr7 i2 v2) i3)))
(assert (= (select (store arr8 i4 v3) i5) (select (store arr9 i1 v1) i2)))

; More assertions with array operations
(assert (>= (select arr10 i3) (select arr11 i4)))
(assert (<= (select arr12 i5) (select arr13 i1)))
(assert (not (= (select arr14 i2) (select arr1 i3))))

(check-sat)