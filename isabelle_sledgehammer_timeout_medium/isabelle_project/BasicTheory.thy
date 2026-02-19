theory BasicTheory
imports Main
begin

(* This theory file contains basic algebraic properties 
   for natural numbers and lists *)

(* Distributivity property for multiplication over addition *)
lemma distributivity_law:
  "(a::nat) * (b + c) = a * b + a * c"
  by sledgehammer

(* Associativity of addition for natural numbers *)
lemma associativity_thm:
  "(a::nat) + (b + c) = (a + b) + c"
  by sledgehammer

(* Commutativity of multiplication *)
lemma multiplication_comm:
  "(a::nat) * b = b * a"
  by sledgehammer

(* List append associativity property *)
lemma list_append_assoc:
  "(xs @ ys) @ zs = xs @ (ys @ (zs::nat list))"
  by sledgehammer

(* Identity property for addition *)
lemma addition_identity:
  "(a::nat) + 0 = a"
  by sledgehammer

(* Length of reversed list equals original length *)
lemma reverse_length:
  "length (rev xs) = length (xs::nat list)"
  by sledgehammer

end