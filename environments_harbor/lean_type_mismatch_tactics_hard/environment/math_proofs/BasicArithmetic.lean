import Mathlib.Data.Nat.Basic

namespace BasicArithmetic

-- Theorem 1: Addition is commutative
theorem add_comm (n m : Nat) : n + m = m + n := by
  induction n with
  | zero => 
    simp
    rfl
  | succ n ih => 
    calc succ n + m = succ (n + m) := by rfl
      _ = succ (n + m) := by rw [ih]
      _ = m + succ n := by rw [Nat.add_succ]

-- Theorem 2: Zero is the right identity for addition
theorem add_zero (n : Nat) : n + 0 = n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    calc succ n + 0 = succ (n + 0) := by rfl
      _ = succ n := by rw [←ih]

-- Theorem 3: Associativity of addition
theorem add_assoc (n m k : Nat) : (n + m) + k = n + (m + k) := by
  induction n with
  | zero =>
    simp
    rfl
  | succ n ih =>
    calc (succ n + m) + k = succ (n + m) + k := by rfl
      _ = succ ((n + m) + k) := by rfl
      _ = succ (n + (m + k)) := by rw [←ih]
      _ = succ n + (m + k) := by rfl

end BasicArithmetic