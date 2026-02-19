(set-logic QF_ALIA)

(declare-const arr1 (Array Int Int))
(declare-const arr2 (Array Int Int))
(declare-const arr3 (Array Int Int))

(assert (= (select (store (select (store (select (store (select (store (select arr1 0) 1 2) 2) 3 4) 4) 5 6) 6) 7 8) 8) 42))

(assert (= (select (store (select (store (select (store (select (store (select (store arr2 10) 11 20) 12) 13 30) 14) 15 40) 16) 17 50) 18) 100))

(assert (> (select (store (select (store (select (store (select (store (select (store (select arr3 5) 6 7) 8) 9 10) 11) 12 13) 14) 15 16) 17) 18 19) 20) 0))

(check-sat)