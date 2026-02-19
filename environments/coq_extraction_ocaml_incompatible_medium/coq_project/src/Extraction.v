(* Extraction Configuration for Data Structures Project *)
(* This file configures the extraction of Coq code to OCaml *)

Require Import ExtrOcamlBasic ExtrOcamlNatInt ExtrOcamlString.
Require Import DataStructures.

(* Extract basic Coq types to OCaml equivalents *)

(* Boolean extraction - standard mapping *)
Extract Inductive bool => "bool" ["true" "false"].

(* Natural number extraction - problematic syntax for successor *)
Extract Inductive nat => "int" ["0" "(fun x -> x + 1)"]
  "(fun fO fS n -> if n=0 then fO () else fS (n-1))".

(* List extraction - standard mapping *)
Extract Inductive list => "list" ["[]" "(::)"].

(* Option extraction - standard mapping *)
Extract Inductive option => "option" ["None" "Some"].

(* Product/pair extraction - may cause tuple issues *)
Extract Inductive prod => "(*)" ["(,)"].

(* Unit extraction *)
Extract Inductive unit => "unit" ["()"].

(* Sum type extraction - potential variant mismatch *)
Extract Inductive sum => "either" ["Left" "Right"].

(* Constants that might have type mismatches *)

(* Arithmetic operations with potential type issues *)
Extract Constant plus => "( + )".
Extract Constant mult => "( * )".
Extract Constant minus => "( - )".

(* Comparison operations - might return wrong type *)
Extract Constant Nat.eqb => "(fun n m -> n = m)".
Extract Constant Nat.leb => "(fun n m -> n <= m)".
Extract Constant Nat.ltb => "(fun n m -> n < m)".

(* List operations with potential arity mismatches *)
Extract Constant List.length => "List.length".
Extract Constant List.app => "List.append".
Extract Constant List.rev => "List.rev".
Extract Constant List.map => "List.map".
Extract Constant List.fold_left => "List.fold_left".

(* String operations *)
Extract Constant String.eqb => "(fun s1 s2 -> s1 = s2)".

(* Custom data structure operations that may have signature mismatches *)
Extract Constant tree_height => 
  "(fun t -> let rec height = function | Leaf -> 0 | Node (l, _, r) -> 1 + max (height l) (height r) in height t)".

(* This constant extraction might cause a type mismatch *)
Extract Constant max => "max".
Extract Constant min => "min".

(* Potentially problematic extraction of comparison result *)
Extract Inductive comparison => "int" ["(-1)" "0" "1"].

(* Custom types from DataStructures that might not align with OCaml *)
(* These might cause tuple/record structure mismatches *)
Extract Constant empty_tree => "Leaf".
Extract Constant singleton => "(fun x -> Node (Leaf, x, Leaf))".

(* This might create issues with currying/uncurrying *)
Extract Constant fold_tree => 
  "(fun f acc tree -> let rec fold acc = function | Leaf -> acc | Node (l, v, r) -> f (fold (fold acc l) r) v in fold acc tree)".

(* Perform the extraction *)
Recursive Extraction Library DataStructures.