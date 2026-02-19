(* DataStructures.v - Verified data structures in Coq *)

Require Import Arith.
Require Import List.
Require Import Omega.

Module DataStructures.

(* Binary tree definition *)
Inductive tree : Type :=
  | Leaf : tree
  | Node : nat -> tree -> tree -> tree.

(* Option type for search results *)
Inductive option (A : Type) : Type :=
  | Some : A -> option A
  | None : option A.

Arguments Some {A} _.
Arguments None {A}.

(* Maybe type for demonstration *)
Inductive maybe (A : Type) : Type :=
  | Just : A -> maybe A
  | Nothing : maybe A.

Arguments Just {A} _.
Arguments Nothing {A}.

(* Size function - counts nodes *)
Fixpoint size (t : tree) : nat :=
  match t with
  | Leaf => 0
  | Node _ left right => 1 + size left + size right
  end.

(* Height function - depth of tree *)
Fixpoint height (t : tree) : nat :=
  match t with
  | Leaf => 0
  | Node _ left right => 1 + max (height left) (height right)
  end.

(* Search function - finds a value in tree *)
Fixpoint search (x : nat) (t : tree) : bool :=
  match t with
  | Leaf => false
  | Node v left right =>
      if Nat.eqb x v then true
      else if Nat.ltb x v then search x left
      else search x right
  end.

(* Insert function - adds value to tree *)
Fixpoint insert (x : nat) (t : tree) : tree :=
  match t with
  | Leaf => Node x Leaf Leaf
  | Node v left right =>
      if Nat.ltb x v then Node v (insert x left) right
      else if Nat.ltb v x then Node v left (insert x right)
      else t
  end.

(* Map function over tree values *)
Fixpoint tree_map (f : nat -> nat) (t : tree) : tree :=
  match t with
  | Leaf => Leaf
  | Node v left right => Node (f v) (tree_map f left) (tree_map f right)
  end.

(* Count leaves in tree *)
Fixpoint count_leaves (t : tree) : nat :=
  match t with
  | Leaf => 1
  | Node _ left right => count_leaves left + count_leaves right
  end.

(* Find minimum value *)
Fixpoint find_min (t : tree) : option nat :=
  match t with
  | Leaf => None
  | Node v Leaf _ => Some v
  | Node v left _ => find_min left
  end.

(* Convert tree to list *)
Fixpoint tree_to_list (t : tree) : list nat :=
  match t with
  | Leaf => nil
  | Node v left right => tree_to_list left ++ (v :: tree_to_list right)
  end.

(* Lemma: size of empty tree is 0 *)
Lemma size_leaf : size Leaf = 0.
Proof.
  simpl.
  reflexivity.
Qed.

(* Lemma: height of empty tree is 0 *)
Lemma height_leaf : height Leaf = 0.
Proof.
  simpl.
  reflexivity.
Qed.

(* Lemma: size increases after insert *)
Lemma insert_increases_size : forall x t,
  size (insert x t) >= size t.
Proof.
  intros x t.
  induction t.
  - simpl. omega.
  - simpl. destruct (Nat.ltb x n) eqn:Hlt.
    + simpl. omega.
    + destruct (Nat.ltb n x) eqn:Hlt2.
      * simpl. omega.
      * omega.
Qed.

(* Lemma: search finds inserted element *)
Lemma search_insert_found : forall x t,
  search x (insert x t) = true.
Proof.
  intros x t.
  induction t.
  - simpl. rewrite Nat.eqb_refl. reflexivity.
  - simpl. destruct (Nat.ltb x n) eqn:Hlt.
    + simpl. rewrite Hlt. apply IHt1.
    + destruct (Nat.ltb n x) eqn:Hlt2.
      * simpl. destruct (Nat.eqb x n) eqn:Heq.
        -- reflexivity.
        -- destruct (Nat.ltb x n) eqn:Hlt3.
           ++ discriminate.
           ++ apply IHt2.
      * simpl. rewrite Nat.eqb_refl. reflexivity.
Qed.

(* Lemma: size is always positive for non-empty trees *)
Lemma size_node_positive : forall v l r,
  size (Node v l r) > 0.
Proof.
  intros v l r.
  simpl.
  omega.
Qed.

(* Lemma: height increases with insert *)
Lemma height_insert : forall x t,
  height (insert x t) >= height t.
Proof.
  intros x t.
  induction t.
  - simpl. omega.
  - simpl. destruct (Nat.ltb x n) eqn:Hlt.
    + simpl. apply Nat.max_le_compat_r. apply IHt1.
    + destruct (Nat.ltb n x) eqn:Hlt2.
      * simpl. apply Nat.max_le_compat_l. apply IHt2.
      * omega.
Qed.

(* Theorem: map preserves structure *)
Theorem map_preserves_size : forall f t,
  size (tree_map f t) = size t.
Proof.
  intros f t.
  induction t.
  - simpl. reflexivity.
  - simpl. rewrite IHt1. rewrite IHt2. reflexivity.
Qed.

(* Lemma: leaves count property *)
Lemma count_leaves_positive : forall t,
  count_leaves t > 0.
Proof.
  intros t.
  induction t.
  - simpl. omega.
  - simpl. omega.
Qed.

(* Lemma: relationship between leaves and nodes *)
Lemma leaves_nodes_relation : forall t,
  count_leaves t = size t + 1.
Proof.
  intros t.
  induction t.
  - simpl. reflexivity.
  - simpl. rewrite IHt1. rewrite IHt2. omega.
Qed.

(* Helper function for testing *)
Definition singleton (x : nat) : tree :=
  Node x Leaf Leaf.

(* Lemma: singleton size is 1 *)
Lemma singleton_size : forall x,
  size (singleton x) = 1.
Proof.
  intros x.
  unfold singleton.
  simpl.
  reflexivity.
Qed.

(* Lemma: singleton height is 1 *)
Lemma singleton_height : forall x,
  height (singleton x) = 1.
Proof.
  intros x.
  unfold singleton.
  simpl.
  reflexivity.
Qed.

End DataStructures.