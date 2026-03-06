(set-logic QF_LIA)

(declare-const viewport_width Int)
(declare-const viewport_height Int)
(declare-const max_texture_size Int)

(assert (! (>= viewport_width 1024) :named c1))
(assert (! (<= viewport_width 512) :named c2))
(assert (! (>= viewport_height 768) :named c3))
(assert (! (<= max_texture_size 4096) :named c4))
(assert (! (>= max_texture_size 256) :named c5))
(assert (! (= viewport_height (* 3 (div viewport_width 4))) :named c6))

(check-sat)