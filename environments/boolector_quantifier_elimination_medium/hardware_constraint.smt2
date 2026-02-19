(set-logic BV)
(declare-const current (_ BitVec 2))
(declare-const next (_ BitVec 2))
(assert
  (exists ((intermediate (_ BitVec 2)))
    (and
      (= intermediate (bvadd current #b01))
      (= next intermediate)
      (bvule current #b10))))
(check-sat)
(exit)