(set-logic QF_LIA)

(declare-const vertex_count Int)
(declare-const batch_size Int)
(declare-const max_vertices Int)
(declare-const processing_threads Int)

(assert (! (= batch_size 10) :named c1))
(assert (! (= (mod vertex_count batch_size) 0) :named c2))
(assert (! (< vertex_count 5) :named c3))
(assert (! (> vertex_count max_vertices) :named c4))
(assert (! (= max_vertices 1000) :named c5))
(assert (! (>= processing_threads 1) :named c6))
(assert (! (<= processing_threads 8) :named c7))

(check-sat)