{-# OPTIONS --safe --sized-types #-}

module Stream where

open import Size
open import Data.Nat

data Stream (A : Set) : Size → Set where
  _::_ : ∀ {i} → A → Stream A i → Stream A (↑ i)

head : ∀ {A i} → Stream A (↑ i) → A
head (x :: xs) = x

tail : ∀ {A i} → Stream A (↑ i) → Stream A i
tail (x :: xs) = xs

map : ∀ {A B i} → (A → B) → Stream A i → Stream B i
map f (x :: xs) = f x :: map f xs

zipWith : ∀ {A B C i} → (A → B → C) → Stream A i → Stream B i → Stream C i
zipWith f (x :: xs) (y :: ys) = f x y :: zipWith f xs ys