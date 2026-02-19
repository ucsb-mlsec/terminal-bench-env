#!/usr/bin/env python3

import sys
import os
from collections import deque

def parse_model(filename):
    """Parse the model file to extract initial state and transitions."""
    initial_state = None
    transitions = []
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('INITIAL:'):
                    # Extract the initial state
                    state_str = line.split('INITIAL:')[1].strip()
                    initial_state = eval(state_str)
                elif line.startswith('TRANSITION:'):
                    # Extract transition
                    trans_str = line.split('TRANSITION:')[1].strip()
                    parts = trans_str.split('->')
                    from_state = eval(parts[0].strip())
                    to_state = eval(parts[1].strip())
                    transitions.append((from_state, to_state))
    except Exception as e:
        print(f"Error parsing model file: {e}")
        return None, []
    
    return initial_state, transitions

def build_transition_map(transitions):
    """Build a map from states to their successor states."""
    trans_map = {}
    for from_state, to_state in transitions:
        if from_state not in trans_map:
            trans_map[from_state] = []
        trans_map[from_state].append(to_state)
    return trans_map

def explore_state_space(initial_state, trans_map):
    """
    Explore the state space starting from initial state.
    Returns the set of visited states.
    
    BUG: Using list instead of set for visited states!
    """
    # BUG 1: Using list instead of set - will cause duplicates
    visited = []
    queue = deque([initial_state])
    
    while queue:
        current_state = queue.popleft()
        
        # BUG 2: Adding to visited AFTER checking transitions
        # This means we might add the same state multiple times
        
        # Get successors for current state
        if current_state in trans_map:
            successors = trans_map[current_state]
            for next_state in successors:
                # BUG 3: Missing check if next_state already visited before adding to queue
                queue.append(next_state)
                visited.append(next_state)
        
        # Add current state to visited (too late!)
        visited.append(current_state)
    
    return visited

def count_unique_states(visited):
    """Count unique states from visited list."""
    # BUG 4: Off-by-one error - subtracting 1 for some incorrect reason
    # Maybe developer thought initial state was counted twice?
    unique = set(visited)
    return len(unique) - 1

def find_deadlocks(states, trans_map):
    """Find states that have no outgoing transitions (deadlocks)."""
    deadlocks = []
    unique_states = set(states)
    
    for state in unique_states:
        # BUG 5: Incorrect deadlock detection - checking if state NOT in map
        # OR if it has successors (should be AND, or just first condition)
        if state not in trans_map or len(trans_map.get(state, [])) > 0:
            deadlocks.append(state)
    
    return deadlocks

def save_results(states_count, deadlocks_count, status):
    """Save exploration results to file."""
    try:
        with open('exploration_results.txt', 'w') as f:
            f.write(f"STATES={states_count}\n")
            f.write(f"DEADLOCKS={deadlocks_count}\n")
            f.write(f"STATUS={status}\n")
        return True
    except Exception as e:
        print(f"Error saving results: {e}")
        return False

def main():
    """Main exploration routine."""
    model_file = 'data/model.txt'
    
    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}")
        save_results(0, 0, "FAIL")
        return
    
    print("Parsing model file...")
    initial_state, transitions = parse_model(model_file)
    
    if initial_state is None:
        print("Failed to parse model file")
        save_results(0, 0, "FAIL")
        return
    
    print(f"Initial state: {initial_state}")
    print(f"Total transitions defined: {len(transitions)}")
    
    # Build transition map
    trans_map = build_transition_map(transitions)
    print(f"States with outgoing transitions: {len(trans_map)}")
    
    # Explore state space
    print("Exploring state space...")
    visited = explore_state_space(initial_state, trans_map)
    
    print(f"Visited states (with duplicates): {len(visited)}")
    
    # Count unique states (buggy)
    unique_count = count_unique_states(visited)
    print(f"Unique states (buggy count): {unique_count}")
    
    # Find deadlocks (buggy)
    deadlocks = find_deadlocks(visited, trans_map)
    deadlock_count = len(deadlocks)
    print(f"Deadlock states found: {deadlock_count}")
    
    # Save results
    status = "PASS"
    if save_results(unique_count, deadlock_count, status):
        print(f"Results saved to exploration_results.txt")
    else:
        status = "FAIL"
        print("Failed to save results")
    
    return unique_count, deadlock_count

if __name__ == "__main__":
    main()