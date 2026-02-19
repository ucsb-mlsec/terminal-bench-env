(set-logic QF_ALIA)

(declare-const arr1 (Array Int Int))
(declare-const arr2 (Array Int Int))
(declare-const arr3 (Array Int Int))
(declare-const arr4 (Array Int Int))
(declare-const arr5 (Array Int Int))
(declare-const arr6 (Array Int Int))
(declare-const arr7 (Array Int Int))
(declare-const arr8 (Array Int Int))
(declare-const arr9 (Array Int Int))
(declare-const arr10 (Array Int Int))
(declare-const arr11 (Array Int Int))
(declare-const arr12 (Array Int Int))
(declare-const arr13 (Array Int Int))

(assert (= (select arr1 0) 10))
(assert (= (select arr2 1) 20))
(assert (= (select arr3 2) 30))
(assert (= (select arr4 3) 40))
(assert (= (select arr5 4) 50))

(assert (= (select (store arr6 0 100) 0) 100))
(assert (= (select (store arr7 1 200) 1) 200))
(assert (= (select (store arr8 2 300) 2) 300))

(assert (> (select arr9 5) (select arr10 5)))
(assert (< (select arr11 6) (select arr12 6)))

(assert (= (select arr13 7) 70))

(assert (= (select (store arr1 10 15) 10) 15))
(assert (= (select (store arr2 11 25) 11) 25))
(assert (= (select (store arr3 12 35) 12) 35))

(assert (>= (select arr4 8) 0))
(assert (<= (select arr5 9) 100))

(assert (= (select arr6 14) (select arr7 14)))
(assert (= (select arr8 15) (select arr9 15)))
(assert (= (select arr10 16) (select arr11 16)))

(assert (not (= (select arr12 20) (select arr13 20))))

(check-sat)