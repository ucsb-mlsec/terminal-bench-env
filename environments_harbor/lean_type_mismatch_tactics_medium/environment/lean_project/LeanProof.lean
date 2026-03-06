-- LeanProof.lean
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic

-- Theorem about list operations with intentional type mismatch errors
theorem list_map_length (xs : List Nat) (f : Nat → Nat) : 
  List.length (List.map f xs) = List.length xs := by
  induction xs with
  | nil => 
    -- Error 1: exact expects Nat but gets Bool
    exact true
  | cons head tail ih =>
    simp [List.map, List.length]
    -- Error 2: Using ih which has correct type, but wrapping in wrong constructor
    exact Nat.succ ih

-- Helper lemma with type errors
lemma append_length (xs ys : List Nat) :
  List.length (xs ++ ys) = List.length xs + List.length ys := by
  induction xs with
  | nil =>
    simp [List.append, List.length]
    -- Error 3: rfl expects equal types but left side is manipulated incorrectly
    exact Nat.zero_add (List.length ys)
  | cons head tail ih =>
    simp [List.append, List.length]
    rw [ih]
    -- Error 4: applying add_comm with wrong number of arguments
    apply Nat.add_comm (List.length tail)

-- Theorem about filtering lists
theorem filter_length_le (xs : List Nat) (p : Nat → Bool) :
  List.length (List.filter p xs) ≤ List.length xs := by
  induction xs with
  | nil =>
    simp [List.filter, List.length]
    -- Error 5: exact expects a proof of ≤ but gets Nat
    exact 0
  | cons head tail ih =>
    simp [List.filter]
    cases p head with
    | false =>
      simp [List.length]
      apply Nat.le_succ_of_le
      exact ih
    | true =>
      simp [List.length]
      apply Nat.succ_le_succ
      -- Error 6: ih has correct type but wrapped incorrectly
      exact Nat.le.refl ih

-- Additional theorem with more type errors
theorem reverse_length (xs : List Nat) :
  List.length (List.reverse xs) = List.length xs := by
  induction xs with
  | nil =>
    rfl
  | cons head tail ih =>
    simp [List.reverse]
    rw [List.length_append]
    simp [List.length]
    -- Error 7: trying to prove equality by providing wrong type
    exact ih + 1

-- Final theorem combining operations
theorem map_append (xs ys : List Nat) (f : Nat → Nat) :
  List.map f (xs ++ ys) = List.map f xs ++ List.map f ys := by
  induction xs with
  | nil =>
    simp [List.append, List.map]
  | cons head tail ih =>
    simp [List.append, List.map]
    -- Error 8: congrArg expects function but type is mismatched
    apply congrArg (List.cons (f head))
    exact ih