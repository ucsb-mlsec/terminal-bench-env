Prover9 SOS Strategy Benchmark Data
==========================================

This directory contains performance benchmark data from a systematic evaluation
of different Set of Support (SOS) strategy configurations in Prover9, an
automated theorem prover. The goal of this benchmark is to identify which
SOS strategy configurations provide the best overall performance across a
diverse set of theorem proving problems.


DATA COLLECTION
---------------

The benchmark was conducted as follows:

- 5 different SOS strategy configurations were tested (config_1 through config_5)
- Each configuration was run against the same set of 10 theorem proving problems
- This resulted in a total of 50 benchmark runs (5 configs × 10 problems)
- Each run was subject to a 60-second timeout limit
- A 2GB memory limit was enforced for all runs
- All tests were conducted on identical hardware to ensure fair comparison


FILE DESCRIPTIONS
-----------------

run_results.csv:
  Contains performance metrics for all 50 benchmark runs with the following columns:
  - run_id: Unique identifier for each benchmark run
  - config_id: Identifier for the SOS strategy configuration used (config_1 to config_5)
  - problem_id: Identifier for the test problem (problem_1 to problem_10)
  - status: Outcome of the run (success, timeout, or memory_error)
  - time_seconds: Completion time in seconds (only populated for successful runs)

problem_metadata.json:
  Contains descriptive information about each of the 10 test problems, including
  problem difficulty, domain, and characteristics relevant to SOS strategy selection.


CONFIGURATION DETAILS
---------------------

The five configurations tested represent different approaches to SOS strategy:

- config_1: Default SOS strategy with standard clause selection heuristics
- config_2: Aggressive clause selection prioritizing shorter clauses
- config_3: Balanced approach with moderate resonance and weight limits
- config_4: Conservative strategy emphasizing completeness over speed
- config_5: Experimental configuration with dynamic SOS adjustments

Note: Lower configuration numbers do not indicate superior performance.
Each configuration has different strengths depending on problem characteristics.


STATUS CODES
------------

success: The prover successfully found a proof within time and memory limits
timeout: The run exceeded the 60-second time limit without completing
memory_error: The run exceeded the 2GB memory limit before completion


NOTES
-----

- The time_seconds field is only recorded for runs with status "success"
- Failed runs (timeout or memory_error) have empty time_seconds values
- For performance analysis, consider both success rate and average completion time
- Success rate is the primary metric; completion time is the secondary tiebreaker
- Some problems may be inherently more difficult and could timeout across all configs
- The most robust configuration will have high success rate with reasonable completion times