# Turing Machine Specification

## Overview

This document specifies a deterministic Turing machine model for binary palindrome recognition. The machine operates on an infinite tape and uses a finite state controller to process input strings composed of binary digits (0 and 1). The machine must determine whether the input string reads the same forwards and backwards.

## Formal Definition

A Turing machine is formally defined as a 7-tuple M = (Q, Γ, Σ, δ, q₀, qₐ, qᵣ) where:

- **States (Q)**: A finite set of states that control the machine's behavior
  - Must include at least: initial state, accept state, and reject state
  - Additional working states as needed for the algorithm

- **Tape Alphabet (Γ)**: {0, 1, _, X} where:
  - '0' and '1' are input symbols representing binary digits
  - '_' (underscore) represents blank cells on the tape
  - 'X' is a marker symbol used for working memory and marking processed symbols

- **Input Alphabet (Σ)**: {0, 1}
  - Subset of Γ consisting of valid input symbols
  - Σ ⊂ Γ

- **Transition Function (δ)**: Q × Γ → Q × Γ × {L, R}
  - Maps (current_state, read_symbol) to (new_state, write_symbol, direction)
  - Must be defined for all non-halting state and symbol combinations that can be reached
  - Deterministic: exactly one transition per (state, symbol) pair
  - Direction: L (move head left), R (move head right)

- **Initial State (q₀)**: The starting state of the machine
  - Machine begins execution in this state

- **Accept State (qₐ)**: Halting state indicating acceptance
  - When reached, the machine halts and accepts the input
  - Conventionally named "qaccept" or "q_accept"

- **Reject State (qᵣ)**: Halting state indicating rejection
  - When reached, the machine halts and rejects the input
  - Conventionally named "qreject" or "q_reject"

## Tape Model

The Turing machine operates on an infinite tape with the following properties:

- **Structure**: The tape extends infinitely in both left and right directions
- **Cells**: Each cell contains exactly one symbol from the tape alphabet Γ
- **Initial Configuration**: 
  - Input string is written on consecutive cells
  - All cells to the left and right of the input contain blanks ('_')
  - Format: `_...___input_string___..._`
- **Head Position**: 
  - A read/write head is positioned over one cell at a time
  - Initial head position is at the leftmost character of the input
  - For empty input, head starts on a blank cell
- **Operations**: In each step, the head can:
  - Read the symbol at its current position
  - Write a new symbol (possibly the same) to its current position
  - Move one cell left (L) or one cell right (R)

## Execution Model

The Turing machine executes according to the following operational semantics:

1. **Initialization**:
   - Machine enters the initial state q₀
   - Tape head is positioned at the leftmost input symbol (or leftmost blank if input is empty)
   - Tape contains the input string surrounded by blanks

2. **Execution Step**: At each computational step:
   - Read the symbol σ currently under the tape head
   - Lookup the transition δ(current_state, σ) = (new_state, write_symbol, direction)
   - Write the write_symbol to the current tape cell (replacing σ)
   - Move the tape head one position in the specified direction (L or R)
   - Transition to the new_state
   - Increment step counter

3. **Halting**:
   - Machine halts when it enters the accept state (qₐ) or reject state (qᵣ)
   - No transitions are defined for halting states
   - The machine's answer is determined by which halting state is reached

4. **Non-termination Prevention**:
   - A step limit (100 steps) is enforced per input
   - If the limit is exceeded, execution is terminated and treated as rejection
   - Well-designed machines should halt well before this limit

## Palindrome Recognition Algorithm (High-Level)

A standard algorithm for palindrome recognition using a Turing machine follows this strategy:

### Algorithm Overview

The machine works by comparing symbols from the outside-in, marking symbols as they are checked:

1. **Mark and Remember Left**:
   - Position head at leftmost unmarked symbol
   - If only blanks and marks remain, accept (palindrome confirmed)
   - Remember whether the symbol is '0' or '1'
   - Mark it with 'X' to indicate it has been processed

2. **Move to Right End**:
   - Move head right, passing over all unmarked symbols
   - Stop at the rightmost unmarked symbol (before blanks or after the last unmarked digit)

3. **Compare and Mark Right**:
   - Check if the rightmost unmarked symbol matches the remembered left symbol
   - If mismatch occurs, reject (not a palindrome)
   - If match, mark the rightmost symbol with 'X'

4. **Return to Left**:
   - Move head back to the left end
   - Position for next iteration

5. **Repeat**:
   - Continue process until all symbols are marked
   - If all comparisons succeed, accept

### State Design Pattern

A typical implementation uses states structured as:
- **q0**: Initial state, scan for leftmost unmarked symbol
- **q1**: Found '0' on left, moving right to find rightmost
- **q2**: Found '1' on left, moving right to find rightmost
- **q3**: Moving right to end after marking left '0'
- **q4**: Moving right to end after marking left '1'
- **q5**: Checking rightmost matches '0'
- **q6**: Checking rightmost matches '1'
- **q7**: Return to left after successful match
- **qaccept**: Accept state (palindrome)
- **qreject**: Reject state (not a palindrome)

### Edge Cases

The algorithm must correctly handle:
- **Empty string**: Should accept (empty string is a palindrome)
- **Single character**: Should accept (any single character is a palindrome)
- **Two identical characters**: Should accept ("00", "11")
- **Two different characters**: Should reject ("01", "10")
- **Odd-length palindromes**: Middle character matches itself
- **Even-length palindromes**: All characters have distinct pairs

## Transition Notation

Transitions are specified in the following format:

```
(state, read) → (next_state, write, move)
```

### Example Transitions

```
(q0, 0) → (q1, X, R)    // Read '0' in state q0, mark with X, go to q1, move right
(q0, 1) → (q2, X, R)    // Read '1' in state q0, mark with X, go to q2, move right
(q0, _) → (qaccept, _, R)  // Empty input or all checked, accept
(q1, 0) → (q1, 0, R)    // Skip over unmarked 0s while moving right
(q1, _) → (q3, _, L)    // Reached end, move back left to check
```

## Example Machine: Single-Bit Checker

Here is a minimal Turing machine that accepts strings containing only identical bits:

**States**: {q0, q1, q2, qaccept, qreject}

**Transitions**:
- (q0, _) → (qaccept, _, R)  // Empty string accepted
- (q0, 0) → (q1, X, R)       // Found 0, mark and check rest
- (q0, 1) → (q2, X, R)       // Found 1, mark and check rest
- (q1, 0) → (q1, X, R)       // Continue marking 0s
- (q1, 1) → (qreject, 1, R)  // Found 1 after 0, reject
- (q1, _) → (qaccept, _, R)  // All 0s, accept
- (q2, 1) → (q2, X, R)       // Continue marking 1s
- (q2, 0) → (qreject, 0, R)  // Found 0 after 1, reject
- (q2, _) → (qaccept, _, R)  // All 1s, accept

This simple machine demonstrates the transition structure but does not implement full palindrome checking.

## Requirements

Any valid Turing machine implementation for palindrome recognition must satisfy:

1. **Halting Property**: Must halt on all possible inputs in finite time
2. **Step Limit**: Must not exceed 100 steps per input string
3. **Correctness**: Must correctly identify all palindromes and non-palindromes
4. **Empty String Handling**: Must accept empty string as a valid palindrome
5. **Single Character Handling**: Must accept any single character as a palindrome
6. **Completeness**: Transition function must be defined for all reachable (state, symbol) pairs
7. **Determinism**: Exactly one transition per (state, symbol) combination
8. **Alphabet Compliance**: Only use symbols from Γ = {0, 1, _, X}

## Performance Expectations

For the test suite of 50 binary strings:
- Total execution time must not exceed 30 seconds
- Average time per test case should be under 1 second
- No individual test case should timeout (exceed 100 steps)
- Well-designed machines typically require O(n²) steps for strings of length n