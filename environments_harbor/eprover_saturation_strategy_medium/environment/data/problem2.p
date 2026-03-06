%------------------------------------------------------------------------------
% File     : problem2.p
% Domain   : Graph Theory
% Problem  : Transitivity of Connected Paths
% Version  : 1.0
% English  : If there is a path from node A to node B, and a path from node B 
%            to node C, then there must be a path from node A to node C. This
%            problem establishes the transitivity property of graph connectivity
%            through edge relations and path composition.
%------------------------------------------------------------------------------

% Axiom 1: Direct edge implies path
fof(edge_is_path, axiom,
    ![X,Y]: (edge(X,Y) => path(X,Y))).

% Axiom 2: Path reflexivity - every node has a path to itself
fof(path_reflexive, axiom,
    ![X]: (node(X) => path(X,X))).

% Axiom 3: Path transitivity - if path from X to Y and Y to Z, then X to Z
fof(path_transitive, axiom,
    ![X,Y,Z]: ((path(X,Y) & path(Y,Z)) => path(X,Z))).

% Axiom 4: Given nodes in the graph
fof(nodes_exist, axiom,
    (node(a) & node(b) & node(c))).

% Axiom 5: Edges create specific connections
fof(edge_a_to_b, axiom,
    edge(a,b)).

% Axiom 6: Another edge connection
fof(edge_b_to_c, axiom,
    edge(b,c)).

% Conjecture: There exists a path from a to c
fof(path_exists_a_to_c, conjecture,
    path(a,c)).

%------------------------------------------------------------------------------