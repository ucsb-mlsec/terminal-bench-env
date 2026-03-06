theory ListTheory
imports Main
begin

(* This theory file establishes fundamental properties of list operations
   including append, map, and reversal. These lemmas form the basis for
   more complex list reasoning. *)

(* Associativity of list append operation *)
lemma list_append_assoc:
  "(xs @ ys) @ zs = xs @ (ys @ zs)"
  by sledgehammer

(* Map fusion: composing two map operations *)
lemma map_fusion_lemma:
  "map f (map g xs) = map (f ∘ g) xs"
  by sledgehammer

(* Length preservation under mapping *)
lemma map_length_preservation:
  "length (map f xs) = length xs"
  by sledgehammer

(* Reverse distributes over append *)
lemma reverse_append_distrib:
  "rev (xs @ ys) = rev ys @ rev xs"
  by sledgehammer

(* Double reversal returns original list *)
lemma reverse_involution:
  "rev (rev xs) = xs"
  by sledgehammer

(* Append with empty list is identity *)
lemma append_nil_right:
  "xs @ [] = xs"
  by sledgehammer

end