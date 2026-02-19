{-# OPTIONS --cubical #-}

module PathProperties where

open import Cubical.Foundations.Prelude
open import Cubical.Foundations.Path
open import Cubical.Foundations.GroupoidLaws

-- Path composition associativity theorem
-- This proves that for paths p : w ≡ x, q : x ≡ y, r : y ≡ z
-- we have (p ∙ q) ∙ r ≡ p ∙ (q ∙ r)

path-comp-assoc : {ℓ : Level} {A : Type ℓ} {w x y z : A}
                → (p : w ≡ x) (q : x ≡ y) (r : y ≡ z)
                → (p ∙ q) ∙ r ≡ p ∙ (q ∙ r)
path-comp-assoc {A = A} {w} {x} {y} {z} p q r = {! need to prove !}

-- Helper lemma for working with path composition
private
  comp-filler : {ℓ : Level} {A : Type ℓ} {x y z : A}
              → (p : x ≡ y) (q : y ≡ z)
              → (i j : I) → A
  comp-filler {x = x} p q i j =
    hfill (λ k → λ { (i = i0) → x
                    ; (i = i1) → q j })
          (inS (p i))
          j

-- Additional properties that might be useful
path-comp-unit-left : {ℓ : Level} {A : Type ℓ} {x y : A}
                    → (p : x ≡ y)
                    → refl ∙ p ≡ p
path-comp-unit-left p = lUnit p

path-comp-unit-right : {ℓ : Level} {A : Type ℓ} {x y : A}
                     → (p : x ≡ y)
                     → p ∙ refl ≡ p
path-comp-unit-right p = rUnit p

-- The module demonstrates path properties in cubical type theory
-- but the main associativity theorem is incomplete