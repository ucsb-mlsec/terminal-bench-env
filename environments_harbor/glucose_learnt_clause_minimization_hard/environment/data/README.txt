SAT Solver Clause Database Log Format
==============================================

This file documents the format of the clause_db.log file, which tracks clause learning events from a SAT solver during its execution.

Overview
--------
The clause database log records every clause that was learned during the solving process. Each learned clause is logged with metadata about its properties and whether it was retained in the clause database after cleanup operations.

File Format
-----------
The log file uses a tab-separated format with 5 fields per line:

timestamp_ms	clause_id	lbd_score	clause_length	retained

Field Descriptions
------------------

1. timestamp_ms (integer)
   - Millisecond timestamp when the clause was learned
   - Represents time elapsed since solver started
   - Values are monotonically increasing

2. clause_id (integer)
   - Unique identifier assigned to each learned clause
   - Sequential numbering starting from some base value
   - No two clauses share the same ID

3. lbd_score (integer, range 1-20)
   - Literal Block Distance score - a quality metric for learned clauses
   - Lower scores indicate higher quality clauses
   - Range: 1 (best) to 20 (worst)
   - Used by solver to decide which clauses to retain

4. clause_length (integer, range 2-100)
   - Number of literals in the learned clause
   - Minimum length is 2 (binary clause)
   - Maximum length is 100

5. retained (boolean: true/false)
   - Indicates whether the clause was kept in the database after cleanup
   - true: clause was retained and remains in memory
   - false: clause was discarded during database reduction

Example Log Entries
-------------------

1250	42	3	8	true
1340	43	7	15	false
1450	44	2	5	true

In the examples above:
- At 1250ms, clause 42 was learned with LBD=3, length=8, and was retained
- At 1340ms, clause 43 was learned with LBD=7, length=15, and was discarded
- At 1450ms, clause 44 was learned with LBD=2, length=5, and was retained

Database Size Calculation
-------------------------
The database size at any given timestamp is the cumulative count of all retained clauses learned up to and including that timestamp. This is calculated by counting all clauses with retained=true from the beginning of the log up to the timestamp in question.

For performance analysis, tracking how the database size grows over time helps identify memory pressure issues and the effectiveness of clause deletion heuristics.