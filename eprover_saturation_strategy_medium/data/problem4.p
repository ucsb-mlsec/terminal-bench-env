% Problem: Partial Order Properties
% This problem explores basic properties of partial orders and strict orders.
% We prove that the strict order derived from a partial order is irreflexive.
% 
% Domain: Order Theory
% Difficulty: Moderate
% 
% A partial order <= is reflexive, antisymmetric, and transitive.
% The strict order < derived from <= is defined as: x < y iff x <= y and x != y
% We prove that this strict order must be irreflexive (no element is strictly less than itself).

% Axiom 1: Reflexivity of partial order
fof(reflexivity, axiom,
    ![X]: leq(X, X)).

% Axiom 2: Antisymmetry of partial order
fof(antisymmetry, axiom,
    ![X, Y]: ((leq(X, Y) & leq(Y, X)) => X = Y)).

% Axiom 3: Transitivity of partial order
fof(transitivity, axiom,
    ![X, Y, Z]: ((leq(X, Y) & leq(Y, Z)) => leq(X, Z))).

% Axiom 4: Definition of strict order from partial order
fof(strict_order_definition, axiom,
    ![X, Y]: (less(X, Y) <=> (leq(X, Y) & X != Y))).

% Axiom 5: Existence of at least one element in the domain
fof(domain_nonempty, axiom,
    ?[A]: leq(A, A)).

% Conjecture: The strict order is irreflexive
fof(irreflexivity_of_strict_order, conjecture,
    ![X]: ~less(X, X)).