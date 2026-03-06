---- MODULE two_phase_commit ----
EXTENDS TLC, Integers, FiniteSets

CONSTANTS 
    RM,          \* Set of resource managers (participants)
    TMState,     \* Set of transaction manager states
    RMState      \* Set of resource manager states

VARIABLES
    tmState,     \* State of the transaction manager
    rmState,     \* Function mapping each RM to its state
    tmPrepared,  \* Set of RMs from which TM has received "Prepared" 
    msgs         \* Set of all messages sent

vars == <<tmState, rmState, tmPrepared, msgs>>

\* Message constructors
PreparedMsg(rm) == [type |-> "Prepared", rm |-> rm]
CommitMsg == [type |-> "Commit"]
AbortMsg == [type |-> "Abort"]

\* Type invariant
TypeOK ==
    /\ tmState \in {"init", "preparing", "committed", "aborted"}
    /\ rmState \in [RM -> {"working", "prepared", "committed", "aborted"}]
    /\ tmPrepared \subseteq RM
    /\ msgs \subseteq (UNION {{PreparedMsg(r) : r \in RM}, {CommitMsg}, {AbortMsg}})

\* Initial state
Init ==
    /\ tmState = "init"
    /\ rmState = [rm \in RM |-> "working"]
    /\ tmPrepared = {}
    /\ msgs = {}

\* Transaction manager sends prepare request
TMPrepare ==
    /\ tmState = "init"
    /\ tmState' = "preparing"
    /\ UNCHANGED <<rmState, tmPrepared, msgs>>

\* Transaction manager receives prepared message
TMRcvPrepared ==
    /\ tmState = "preparing"
    /\ \E rm \in RM :
        /\ PreparedMsg(rm) \in msgs
        /\ rm \notin tmPrepared
        /\ tmPrepared' = tmPrepared \cup {rm}
        /\ UNCHANGED <<tmState, rmState, msgs>>

\* Transaction manager commits
TMCommit ==
    /\ tmState = "preparing"
    /\ tmPrepared = RM
    /\ tmState' = "committed"
    /\ msgs' = msgs \cup {CommitMsg}
    /\ UNCHANGED <<rmState, tmPrepared>>

\* Transaction manager aborts
TMAbort ==
    /\ tmState \in {"preparing", "init"}
    /\ tmState' = "aborted"
    /\ msgs' = msgs \cup {AbortMsg}
    /\ UNCHANGED <<rmState, tmPrepared>>

\* Resource manager prepares
RMPrepare ==
    \E rm \in RM :
        /\ rmState[rm] = "working"
        /\ tmState = "preparing"
        /\ rmState' = [rmState EXCEPT ![rm] = "prepared"]
        /\ msgs' = msgs \cup {PreparedMsg(rm)}
        /\ UNCHANGED <<tmState, tmPrepared>>

\* Resource manager spontaneously decides to abort
RMChooseToAbort ==
    \E rm \in RM :
        /\ rmState[rm] = "working"
        /\ rmState' = [rmState EXCEPT ![rm] = "aborted"]
        /\ UNCHANGED <<tmState, tmPrepared, msgs>>

\* Resource manager receives commit message
RMRcvCommitMsg ==
    \E rm \in RM :
        /\ CommitMsg \in msgs
        /\ rmState[rm] \in {"working", "prepared"}
        /\ rmState' = [rmState EXCEPT ![rm] = "committed"]
        /\ UNCHANGED <<tmState, tmPrepared, msgs>>

\* Resource manager receives abort message  
RMRcvAbortMsg ==
    \E rm \in RM :
        /\ AbortMsg \in msgs
        /\ rmState[rm] \in {"working", "prepared"}
        /\ rmState' = [rmState EXCEPT ![rm] = "aborted"]
        /\ UNCHANGED <<tmState, tmPrepared, msgs>>

\* Helper predicate using CHOOSE (inefficient)
SomeRMPrepared ==
    \E rm \in RM : 
        /\ rmState[rm] = "prepared"
        /\ rm = CHOOSE r \in RM : rmState[r] = "prepared"

\* Helper predicate using CHOOSE (inefficient)
SomeRMCommitted ==
    \E rm \in RM :
        /\ rmState[rm] = "committed"  
        /\ rm = CHOOSE r \in RM : rmState[r] = "committed"

\* Inefficient check using CHOOSE
CanCommit ==
    /\ tmState = "preparing"
    /\ Cardinality({rm \in RM : rmState[rm] = "prepared"}) = Cardinality(RM)
    /\ \A rm \in RM : rm = CHOOSE r \in tmPrepared : r = rm

\* Inefficient check using set comprehension and CHOOSE
AllPreparedMsgs ==
    /\ Cardinality({m \in msgs : m.type = "Prepared"}) > 0
    /\ \E m \in msgs : 
        /\ m.type = "Prepared"
        /\ m = CHOOSE msg \in msgs : msg.type = "Prepared"

\* Another inefficient pattern
HasCommitMsg ==
    /\ {m \in msgs : m.type = "Commit"} # {}
    /\ CHOOSE m \in msgs : m.type = "Commit" \in msgs

\* Inefficient message check
PreparedRMs ==
    {rm \in RM : \E m \in msgs : 
        /\ m.type = "Prepared" 
        /\ m.rm = rm
        /\ m = CHOOSE msg \in {x \in msgs : x.type = "Prepared"} : msg.rm = rm}

\* Next state relation with multiple disjunctions
Next ==
    \/ TMPrepare
    \/ TMRcvPrepared
    \/ TMCommit  
    \/ TMAbort
    \/ RMPrepare
    \/ RMChooseToAbort
    \/ RMRcvCommitMsg
    \/ RMRcvAbortMsg
    \/ (TMPrepare \/ TMCommit)
    \/ (TMAbort \/ RMChooseToAbort)
    \/ (RMRcvCommitMsg \/ RMRcvAbortMsg)
    \/ (TMRcvPrepared \/ RMPrepare)

\* Specification
Spec == Init /\ [][Next]_vars

\* Consistency property
Consistency ==
    \A rm1, rm2 \in RM :
        \/ rmState[rm1] # "committed"
        \/ rmState[rm2] # "aborted"
        \/ rmState[rm1] = rmState[rm2]

====