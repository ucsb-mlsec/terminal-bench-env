{-# OPTIONS --safe #-}

module CoList where

open import Level

private
  variable
    ℓ : Level
    A B : Set ℓ

data CoListF (A : Set ℓ) (CoList : Set ℓ) : Set ℓ where
  []  : CoListF A CoList
  _∷_ : A → CoList → CoListF A CoList

record CoList (A : Set ℓ) : Set ℓ where
  coinductive
  field
    force : CoListF A (CoList A)

open CoList public

-- Construct a cons cell
_∷ʳ_ : {A : Set ℓ} → A → CoList A → CoList A
force (x ∷ʳ xs) = x ∷ xs

-- Empty colist
[] : {A : Set ℓ} → CoList A
force [] = []

-- Repeat an element infinitely
repeat : {A : Set ℓ} → A → CoList A
force (repeat x) = x ∷ repeat x

-- Head function
head : {A : Set ℓ} → CoList A → CoListF A (CoList A)
head xs = force xs

-- Tail function (partial - only works for non-empty lists)
tail : {A : Set ℓ} → CoList A → CoList A
tail xs with force xs
... | [] = []
... | y ∷ ys = ys

-- Map over colist
map : {A B : Set ℓ} → (A → B) → CoList A → CoList B
force (map f xs) with force xs
... | [] = []
... | y ∷ ys = f y ∷ map f ys

-- Zip two colists
zipWith : {A B C : Set ℓ} → (A → B → C) → CoList A → CoList B → CoList C
force (zipWith f xs ys) with force xs | force ys
... | [] | _ = []
... | _ | [] = []
... | x ∷ xs' | y ∷ ys' = f x y ∷ zipWith f xs' ys'