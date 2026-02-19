module Proof where

open import Data.Nat using (ℕ; zero; suc; _+_; _∸_; _≤_; _<_; z≤n; s≤s)
open import Data.List using (List; []; _∷_; length; filter; map)
open import Data.Bool using (Bool; true; false; if_then_else_)
open import Relation.Nullary using (yes; no)
open import Relation.Binary.Propositional using (_≡_)

-- Helper: Check if a number is even
isEven : ℕ → Bool
isEven zero = true
isEven (suc zero) = false
isEven (suc (suc n)) = isEven n

-- The main function that the termination checker rejects
-- This function processes a list by:
-- 1. If the list is empty, return 0
-- 2. If the first element is even, add it and recurse on the tail
-- 3. If the first element is odd, increment it and recurse on the modified list
--
-- Developer's reasoning: This terminates because whenever we encounter an odd number,
-- we make it even (by incrementing), and then on the next iteration it will be consumed.
-- So we're always making progress through the list. The total "work" is bounded by
-- the initial length plus the number of odd elements, which is finite.
--
-- However, Agda can't see this because when we recurse with (suc x ∷ xs), we're not
-- recursing on a structurally smaller term - we've actually made the first element larger!
-- The termination checker doesn't recognize that we're making progress in a different metric
-- (converting odds to evens, which will then be consumed).

{-# TERMINATING #-}
processOddEven : List ℕ → ℕ
processOddEven [] = zero
processOddEven (x ∷ xs) with isEven x
... | true = x + processOddEven xs
... | false = processOddEven (suc x ∷ xs)

-- Test function to make the code runnable
compute : List ℕ → ℕ
compute = processOddEven