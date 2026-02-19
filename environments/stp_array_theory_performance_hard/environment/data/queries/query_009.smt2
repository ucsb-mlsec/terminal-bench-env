(set-logic QF_ALIA)

(declare-fun arr1 () (Array Int Int))
(declare-fun arr2 () (Array Int Int))
(declare-fun arr3 () (Array Int Int))
(declare-fun arr4 () (Array Int Int))
(declare-fun arr5 () (Array Int Int))
(declare-fun idx () Int)

(assert (= (select arr1 0) 10))
(assert (= (select arr1 1) 20))
(assert (= (select arr2 0) 30))
(assert (= (select arr2 1) 40))
(assert (= (select arr3 0) 50))

(assert (= (select (store arr1 2 (select arr2 0)) 2) 30))
(assert (= (select (store arr2 3 (select arr3 1)) 3) (select arr3 1)))
(assert (= (select (store arr3 4 100) 4) 100))

(assert (= (select arr4 idx) (select arr1 0)))
(assert (= (select arr4 (+ idx 1)) (select arr2 1)))
(assert (= (select arr5 0) (select arr1 1)))
(assert (= (select arr5 1) (select arr2 0)))

(assert (= (select (store (store arr1 5 60) 6 70) 5) 60))
(assert (= (select (store (store arr2 7 80) 8 90) 8) 90))
(assert (= (select (store (store arr3 9 100) 10 110) 10) 110))

(assert (= (select (store arr4 11 (select (store arr1 12 120) 12)) 11) 120))
(assert (= (select (store arr5 13 (select (store arr2 14 140) 14)) 13) 140))

(assert (= (select arr1 15) (select arr2 15)))
(assert (= (select arr2 16) (select arr3 16)))
(assert (= (select arr3 17) (select arr4 17)))
(assert (= (select arr4 18) (select arr5 18)))
(assert (= (select arr5 19) (select arr1 19)))

(assert (= (select (store arr1 20 200) 20) 200))
(assert (= (select (store arr2 21 210) 21) 210))
(assert (= (select (store arr3 22 220) 22) 220))
(assert (= (select (store arr4 23 230) 23) 230))
(assert (= (select (store arr5 24 240) 24) 240))

(assert (= (select (store (store arr1 25 250) 26 260) 26) 260))
(assert (= (select (store (store arr2 27 270) 28 280) 27) 270))

(assert (= (select arr1 29) (select (store arr2 29 290) 29)))
(assert (= (select arr3 30) (select (store arr4 30 300) 30)))
(assert (= (select arr5 31) (select (store arr1 31 310) 31)))

(assert (= (select (store arr1 32 (select arr2 32)) 32) (select arr2 32)))
(assert (= (select (store arr3 33 (select arr4 33)) 33) (select arr4 33)))

(assert (= (select arr1 34) 340))
(assert (= (select arr2 35) 350))
(assert (= (select arr3 36) 360))
(assert (= (select arr4 37) 370))

(check-sat)