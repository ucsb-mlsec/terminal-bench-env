(set-logic QF_NIA)

(declare-const width Int)
(declare-const height Int)
(declare-const memory Int)
(declare-const mipmap_levels Int)

(assert (! (> width 0) :named c1))
(assert (! (> height 0) :named c2))
(assert (! (< memory 1024) :named c3))
(assert (! (> memory 2048) :named c4))
(assert (! (= memory (* width height 4)) :named c5))
(assert (! (and (>= mipmap_levels 1) (<= mipmap_levels 12)) :named c6))
(assert (! (>= width 512) :named c7))
(assert (! (<= (* width height) 262144) :named c8))

(check-sat)