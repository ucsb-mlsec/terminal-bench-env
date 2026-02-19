{-# OPTIONS --safe #-}

module Stream where

open import CoList
open import Nat

-- Infinite streams as coinductive data
record Stream (A : Set) : Set where
  coinductive
  field
    head : A
    tail : Stream A

open Stream

-- Map a function over a stream
map : {A B : Set} → (A → B) → Stream A → Stream B
head (map f s) = f (head s)
tail (map f s) = map f (tail s)

-- Zip two streams with a binary function
zipWith : {A B C : Set} → (A → B → C) → Stream A → Stream B → Stream C
head (zipWith f s1 s2) = f (head s1) (head s2)
tail (zipWith f s1 s2) = zipWith f (tail s1) (tail s2)

-- Interleave two streams: take alternating elements
interleave : {A : Set} → Stream A → Stream A → Stream A
head (interleave s1 s2) = head s1
tail (interleave s1 s2) = interleave s2 (tail s1)

-- Drop n elements from a stream
drop : {A : Set} → Nat → Stream A → Stream A
drop zero s = s
drop (suc n) s = drop n (tail s)

-- Take n elements from a stream and return as a CoList
take : {A : Set} → Nat → Stream A → CoList A
take zero s = []
take (suc n) s = head s ∷ take n (tail s)

-- Repeat a value infinitely
repeat : {A : Set} → A → Stream A
head (repeat x) = x
tail (repeat x) = repeat x

-- Cycle through a non-empty CoList infinitely
cycle : {A : Set} → A → CoList A → Stream A
cycle x [] = repeat x
cycle x (y ∷ ys) = helper y ys x (y ∷ ys)
  where
    helper : {A : Set} → A → CoList A → A → CoList A → Stream A
    head (helper current rest first full) = current
    tail (helper current [] first full) = cycle first full
    tail (helper current (z ∷ zs) first full) = helper z zs first full

-- Unfold a stream from a seed value
unfold : {A B : Set} → (B → A) → (B → B) → B → Stream A
head (unfold f g seed) = f seed
tail (unfold f g seed) = unfold f g (g seed)

-- Diagonal interleaving of streams (harder termination pattern)
merge : {A : Set} → Stream A → Stream A → Stream A
head (merge s1 s2) = head s1
tail (merge s1 s2) = merge s2 (tail s1)