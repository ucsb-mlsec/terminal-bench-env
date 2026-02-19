(set-logic QF_AUFLIA)

; Array variable declarations
(declare-const arr1 (Array Int Int))
(declare-const arr2 (Array Int Int))
(declare-const arr3 (Array Int Int))
(declare-const arr4 (Array Int Int))
(declare-const arr5 (Array Int Int))
(declare-const arr6 (Array Int Int))

; Integer variables for indexing
(declare-const i1 Int)
(declare-const i2 Int)
(declare-const i3 Int)
(declare-const i4 Int)
(declare-const v1 Int)
(declare-const v2 Int)
(declare-const v3 Int)

; Assertions with store operations
(assert (= arr2 (store arr1 0 10)))
(assert (= arr3 (store arr2 1 20)))
(assert (= arr4 (store arr3 2 30)))
(assert (= arr5 (store arr4 3 40)))
(assert (= arr6 (store arr5 4 50)))

; Complex assertions with select operations
(assert (= (select arr1 i1) 100))
(assert (= (select arr2 i2) 200))
(assert (= (select arr3 i3) 300))
(assert (= (select arr4 i4) 400))
(assert (> (select arr1 0) (select arr2 1)))
(assert (< (select arr3 2) (select arr4 3)))
(assert (= (select arr5 4) (select arr6 5)))

; Store operations with computed values
(assert (= arr1 (store arr6 10 (+ v1 v2))))
(assert (= arr2 (store arr1 11 (- v2 v3))))
(assert (= arr3 (store arr2 12 (* v1 2))))
(assert (= arr4 (store arr3 13 (+ v3 5))))
(assert (= arr5 (store arr4 14 (- v1 10))))

; Nested operations with select inside store
(assert (= arr6 (store arr5 15 (select arr1 0))))
(assert (= arr1 (store arr6 16 (select arr2 1))))
(assert (= arr2 (store arr1 17 (select arr3 2))))
(assert (= arr3 (store arr2 18 (select arr4 3))))
(assert (= arr4 (store arr3 19 (select arr5 4))))

; More select operations in constraints
(assert (>= (select arr1 5) 0))
(assert (<= (select arr2 6) 1000))
(assert (= (select arr3 7) (select arr4 8)))
(assert (!= (select arr5 9) (select arr6 10)))
(assert (> (select arr1 11) (select arr2 12)))
(assert (< (select arr3 13) (select arr4 14)))

; Additional store operations
(assert (= arr5 (store arr4 20 60)))
(assert (= arr6 (store arr5 21 70)))
(assert (= arr1 (store arr6 22 80)))
(assert (= arr2 (store arr1 23 90)))
(assert (= arr3 (store arr2 24 100)))

; More complex select expressions
(assert (= (+ (select arr1 25) (select arr2 26)) 500))
(assert (= (- (select arr3 27) (select arr4 28)) 100))
(assert (= (* (select arr5 29) 2) (select arr6 30)))
(assert (> (+ (select arr1 31) (select arr2 32)) 600))
(assert (< (- (select arr3 33) (select arr4 34)) 50))

; Additional array operations
(assert (= (select arr5 35) (select arr1 36)))
(assert (= (select arr6 37) (select arr2 38)))
(assert (= (select arr1 39) (select arr3 40)))
(assert (= (select arr2 41) (select arr4 42)))
(assert (= (select arr3 43) (select arr5 44)))

; Final store operations
(assert (= arr4 (store arr3 45 110)))
(assert (= arr5 (store arr4 46 120)))
(assert (= arr6 (store arr5 47 130)))

; More select constraints
(assert (>= (select arr4 48) 0))
(assert (<= (select arr5 49) 2000))
(assert (= (select arr6 50) (select arr1 51)))
(assert (!= (select arr2 52) (select arr3 53)))
(assert (> (select arr4 54) (select arr5 55)))

(check-sat)