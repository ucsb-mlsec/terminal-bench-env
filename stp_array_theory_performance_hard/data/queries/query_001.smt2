(set-logic QF_AUFLIA)
(declare-const arr1 (Array Int Int))
(declare-const arr2 (Array Int Int))
(declare-const arr3 (Array Int Int))
(declare-const x Int)
(declare-const y Int)

(assert (= (select arr1 0) 5))
(assert (= (select arr1 1) 10))
(assert (= (select arr2 2) 15))
(assert (= (select arr1 3) 20))
(assert (= (select arr2 4) 25))
(assert (= (select arr3 0) 30))

(assert (= (store arr1 0 42) arr2))
(assert (= (store arr2 1 55) arr3))
(assert (= (store arr1 2 (select arr2 0)) arr3))

(assert (= (select (store arr1 5 100) 5) 100))

(assert (> x 0))
(assert (< y 100))

(check-sat)
(exit)