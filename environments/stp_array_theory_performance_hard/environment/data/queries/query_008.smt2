(set-logic QF_ALIA)

(declare-const arr1 (Array Int Int))
(declare-const arr2 (Array Int Int))
(declare-const idx Int)
(declare-const val Int)

(assert (= (select arr1 0) 10))
(assert (= (select arr1 idx) val))
(assert (= (select arr2 5) 20))

(check-sat)