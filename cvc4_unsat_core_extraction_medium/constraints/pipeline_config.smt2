(set-logic QF_LIA)

(declare-const stage1_time Int)
(declare-const stage2_time Int)
(declare-const stage3_time Int)
(declare-const total_time Int)
(declare-const buffer_size Int)
(declare-const throughput Int)

(assert (! (= total_time (+ stage1_time stage2_time stage3_time)) :named c1))
(assert (! (> stage1_time 100) :named c2))
(assert (! (> stage2_time 150) :named c3))
(assert (! (< total_time 200) :named c4))
(assert (! (>= buffer_size 1024) :named c5))
(assert (! (= throughput (* buffer_size 2)) :named c6))
(assert (! (< throughput 2500) :named c7))
(assert (! (>= stage3_time 0) :named c8))

(check-sat)