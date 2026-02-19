(set-logic QF_LIA)

(declare-const red Int)
(declare-const green Int)
(declare-const blue Int)

(assert (! (and (>= red 0) (<= red 255)) :named c1))
(assert (! (and (>= green 0) (<= green 255)) :named c2))
(assert (! (and (>= blue 0) (<= blue 255)) :named c3))
(assert (! (> red 300) :named c4))
(assert (! (> green 80) :named c5))
(assert (! (< (+ green blue) 100) :named c6))

(check-sat)