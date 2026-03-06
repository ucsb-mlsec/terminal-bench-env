----------------------------- MODULE two_phase_commit -----------------------------

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS 
    Nodes,      \* Set of database nodes
    NULL        \* Null value

VARIABLES
    coordState,     \* Coordinator state: "idle", "preparing", "committing", "aborting", "done"
    nodeStates,     \* Function from Nodes to their states
    messages,       \* Set of all messages ever sent (PROBLEM: unbounded accumulation)
    votes,          \* Function from Nodes to their votes
    decision,       \* Final decision: NULL, "commit", or "abort"
    txnId           \* Current transaction ID counter

vars == <<coordState, nodeStates, messages, votes, decision, txnId>>

TypeOK ==
    /\ coordState \in {"idle", "preparing", "committing", "aborting", "done"}
    /\ nodeStates \in [Nodes -> {"working", "prepared", "committed", "aborted"}]
    /\ messages \subseteq [type: {"prepare", "vote_yes", "vote_no", "commit", "abort"},
                          sender: Nodes \cup {"coordinator"},
                          receiver: Nodes \cup {"coordinator"},
                          txn: Nat]
    /\ votes \in [Nodes -> {"yes", "no", NULL}]
    /\ decision \in {"commit", "abort", NULL}
    /\ txnId \in Nat

Init ==
    /\ coordState = "idle"
    /\ nodeStates = [n \in Nodes |-> "working"]
    /\ messages = {}        \* Start with empty message set
    /\ votes = [n \in Nodes |-> NULL]
    /\ decision = NULL
    /\ txnId = 0

\* Coordinator initiates prepare phase
CoordinatorPrepare ==
    /\ coordState = "idle"
    /\ coordState' = "preparing"
    /\ messages' = messages \cup 
                   {[type |-> "prepare", 
                     sender |-> "coordinator", 
                     receiver |-> n,
                     txn |-> txnId] : n \in Nodes}
    /\ txnId' = txnId + 1
    /\ UNCHANGED <<nodeStates, votes, decision>>

\* Node receives prepare and votes
NodeVoteYes(n) ==
    /\ nodeStates[n] = "working"
    /\ \E m \in messages : 
        /\ m.type = "prepare"
        /\ m.receiver = n
        /\ m.txn = txnId - 1
    /\ nodeStates' = [nodeStates EXCEPT ![n] = "prepared"]
    /\ votes' = [votes EXCEPT ![n] = "yes"]
    /\ messages' = messages \cup 
                   {[type |-> "vote_yes",
                     sender |-> n,
                     receiver |-> "coordinator",
                     txn |-> txnId - 1]}
    /\ UNCHANGED <<coordState, decision, txnId>>

NodeVoteNo(n) ==
    /\ nodeStates[n] = "working"
    /\ \E m \in messages : 
        /\ m.type = "prepare"
        /\ m.receiver = n
        /\ m.txn = txnId - 1
    /\ nodeStates' = [nodeStates EXCEPT ![n] = "aborted"]
    /\ votes' = [votes EXCEPT ![n] = "no"]
    /\ messages' = messages \cup 
                   {[type |-> "vote_no",
                     sender |-> n,
                     receiver |-> "coordinator",
                     txn |-> txnId - 1]}
    /\ UNCHANGED <<coordState, decision, txnId>>

\* Coordinator decides based on votes
CoordinatorCommit ==
    /\ coordState = "preparing"
    /\ \A n \in Nodes : votes[n] = "yes"
    /\ coordState' = "committing"
    /\ decision' = "commit"
    /\ UNCHANGED <<nodeStates, messages, votes, txnId>>

CoordinatorAbort ==
    /\ coordState = "preparing"
    /\ \E n \in Nodes : votes[n] = "no"
    /\ coordState' = "aborting"
    /\ decision' = "abort"
    /\ UNCHANGED <<nodeStates, messages, votes, txnId>>

\* Coordinator sends decision to nodes
CoordinatorNotifyCommit ==
    /\ coordState = "committing"
    /\ messages' = messages \cup
                   {[type |-> "commit",
                     sender |-> "coordinator",
                     receiver |-> n,
                     txn |-> txnId - 1] : n \in Nodes}
    /\ coordState' = "done"
    /\ UNCHANGED <<nodeStates, votes, decision, txnId>>

CoordinatorNotifyAbort ==
    /\ coordState = "aborting"
    /\ messages' = messages \cup
                   {[type |-> "abort",
                     sender |-> "coordinator",
                     receiver |-> n,
                     txn |-> txnId - 1] : n \in Nodes}
    /\ coordState' = "done"
    /\ UNCHANGED <<nodeStates, votes, decision, txnId>>

\* Nodes execute the decision
NodeCommit(n) ==
    /\ nodeStates[n] = "prepared"
    /\ \E m \in messages :
        /\ m.type = "commit"
        /\ m.receiver = n
    /\ nodeStates' = [nodeStates EXCEPT ![n] = "committed"]
    /\ UNCHANGED <<coordState, messages, votes, decision, txnId>>

NodeAbort(n) ==
    /\ nodeStates[n] \in {"prepared", "working"}
    /\ \E m \in messages :
        /\ m.type = "abort"
        /\ m.receiver = n
    /\ nodeStates' = [nodeStates EXCEPT ![n] = "aborted"]
    /\ UNCHANGED <<coordState, messages, votes, decision, txnId>>

Next ==
    \/ CoordinatorPrepare
    \/ \E n \in Nodes : NodeVoteYes(n)
    \/ \E n \in Nodes : NodeVoteNo(n)
    \/ CoordinatorCommit
    \/ CoordinatorAbort
    \/ CoordinatorNotifyCommit
    \/ CoordinatorNotifyAbort
    \/ \E n \in Nodes : NodeCommit(n)
    \/ \E n \in Nodes : NodeAbort(n)

Spec == Init /\ [][Next]_vars

\* Safety property: No conflicting decisions among nodes
NoConflictingDecisions ==
    \A n1, n2 \in Nodes :
        (nodeStates[n1] = "committed" /\ nodeStates[n2] = "aborted") => FALSE

=============================================================================