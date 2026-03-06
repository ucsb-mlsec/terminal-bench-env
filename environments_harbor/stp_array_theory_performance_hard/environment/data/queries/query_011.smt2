(set-logic QF_ABV)

; Array variable declarations
(declare-fun arr1 () (Array Int Int))
(declare-fun arr2 () (Array Int Int))
(declare-fun arr3 () (Array Int Int))
(declare-fun arr4 () (Array Int Int))

; Scalar variables for indexing and values
(declare-fun x () Int)
(declare-fun y () Int)
(declare-fun z () Int)

; Assertion with exactly 5 levels of nesting
; Level 1: innermost select
; Level 2: store wrapping level 1
; Level 3: select wrapping level 2
; Level 4: store wrapping level 3
; Level 5: select wrapping level 4
(assert (= (select (store (select (store (select arr1 0) 1 2) 2) 3 4) 4) 5))

; Additional assertions with lower nesting (3 levels)
(assert (= (select (store (select arr2 1) 2 10) 2) 10))

; Assertions with 2 levels of nesting
(assert (= (store (select arr3 x) y 100) (store arr4 y 100)))

; Simple operations with 1 level
(assert (= (select arr1 5) 7))
(assert (= (select arr2 3) 9))

; More operations to reach 20-25 total operations
(assert (> (select arr3 0) 0))
(assert (< (select arr4 1) 100))

; Store operations
(assert (= (select (store arr1 10 20) 10) 20))
(assert (= (select (store arr2 15 30) 15) 30))

; Combined assertions with multiple operations
(assert (= (select (store arr3 x 50) x) 50))
(assert (not (= (select arr1 z) (select arr2 z))))

; Additional simple array operations
(assert (= (select arr4 7) (select arr3 7)))
(assert (>= (select arr1 2) (select arr2 2)))

(check-sat)