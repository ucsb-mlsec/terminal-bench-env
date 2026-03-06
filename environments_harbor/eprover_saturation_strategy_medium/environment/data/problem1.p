% Problem: Transitivity of subset and element relationships
% This problem tests basic set theory reasoning about subsets and membership
% Domain: Set theory with subset and element relationships

% Axiom 1: Transitivity of subset relation
fof(subset_transitive, axiom,
    ![X,Y,Z]: ((subset(X,Y) & subset(Y,Z)) => subset(X,Z))).

% Axiom 2: Element of subset is element of superset
fof(element_subset, axiom,
    ![X,Y,E]: ((element(E,X) & subset(X,Y)) => element(E,Y))).

% Axiom 3: Subset is reflexive
fof(subset_reflexive, axiom,
    ![X]: subset(X,X)).

% Axiom 4: Given facts about specific sets
fof(given_subsets, axiom,
    (subset(a,b) & subset(b,c))).

% Axiom 5: Element relationship
fof(given_element, axiom,
    element(e,a)).

% Conjecture: If e is in a, and a is subset of b, and b is subset of c, then e is in c
fof(goal, conjecture,
    element(e,c)).