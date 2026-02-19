{-# OPTIONS --safe #-}

module Nat where

-- Natural numbers definition
data ℕ : Set where
  zero : ℕ
  suc  : ℕ → ℕ

{-# BUILTIN NATURAL ℕ #-}

-- Addition
_+_ : ℕ → ℕ → ℕ
zero + n = n
suc m + n = suc (m + n)

infixl 6 _+_

-- Multiplication
_*_ : ℕ → ℕ → ℕ
zero * n = zero
suc m * n = n + (m * n)

infixl 7 _*_

-- Predecessor
pred : ℕ → ℕ
pred zero = zero
pred (suc n) = n

-- Subtraction
_∸_ : ℕ → ℕ → ℕ
m ∸ zero = m
zero ∸ suc n = zero
suc m ∸ suc n = m ∸ n

infixl 6 _∸_

-- Even predicate
data Bool : Set where
  true  : Bool
  false : Bool

even : ℕ → Bool
even zero = true
even (suc zero) = false
even (suc (suc n)) = even n

-- Double function
double : ℕ → ℕ
double zero = zero
double (suc n) = suc (suc (double n))

-- Half function
half : ℕ → ℕ
half zero = zero
half (suc zero) = zero
half (suc (suc n)) = suc (half n)