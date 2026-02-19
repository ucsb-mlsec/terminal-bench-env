(set-logic QF_AALIA)

(declare-const arr1 (Array Int Int))
(declare-const arr2 (Array Int Int))
(declare-const arr3 (Array Int Int))
(declare-const arr4 (Array Int Int))

(declare-const i Int)
(declare-const j Int)
(declare-const k Int)
(declare-const v1 Int)
(declare-const v2 Int)

(assert (= i 5))
(assert (= j 10))
(assert (= k 15))

(assert (= v1 (select arr1 i)))
(assert (= v2 (select arr2 j)))

(assert (= (select arr3 k) 100))
(assert (= (select arr4 i) 200))

(assert (= (select (store arr1 i 42) j) 42))
(assert (= (select (store arr2 j v1) k) v1))

(assert (> (select arr1 i) 0))
(assert (< (select arr2 j) 100))
(assert (= (select arr3 k) (select arr4 k)))

(assert (= (select arr1 5) (select arr2 5)))
(assert (= (select arr3 10) (select arr4 10)))

(assert (= (select arr1 i) (select (store arr2 i v2) i)))
(assert (= (select arr3 j) (select (store arr4 j 50) j)))

(assert (not (= (select arr1 i) (select arr2 i))))

(check-sat)