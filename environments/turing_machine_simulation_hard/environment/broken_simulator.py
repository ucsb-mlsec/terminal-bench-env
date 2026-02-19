#!/usr/bin/env python3
"""
Broken Turing Machine Simulator
A deliberately buggy implementation of a Turing machine for binary palindrome validation.
Contains several bugs that need to be identified and fixed.
"""

import json
from typing import List, Dict, Optional, Tuple
import sys


class TuringMachine:
    """
    A Turing machine simulator with intentional bugs.
    
    The Turing machine operates on a tape with symbols from a finite alphabet.
    It has a read/write head that can move left or right, and transitions
    between states based on the current state and symbol read.
    """
    
    def __init__(self, states: List[str], initial_state: str, 
                 accept_state: str, reject_state: str, 
                 transitions: List[Dict]):
        """
        Initialize the Turing machine.
        
        Args:
            states: List of all possible states
            initial_state: The starting state
            accept_state: State that accepts input
            reject_state: State that rejects input
            transitions: List of transition rules
        """
        self.states = states
        self.initial_state = initial_state
        self.accept_state = accept_state
        self.reject_state = reject_state
        self.transition_map = self.load_transitions(transitions)
        
        # Tape and machine state
        self.tape: List[str] = []
        self.head_position: int = 0
        self.current_state: str = initial_state
        self.step_count: int = 0
        
    def load_transitions(self, transition_list: List[Dict]) -> Dict[Tuple[str, str], Tuple[str, str, str]]:
        """
        Convert transition list to a lookup dictionary.
        
        Args:
            transition_list: List of transition dictionaries
            
        Returns:
            Dictionary mapping (state, symbol) -> (next_state, write_symbol, direction)
        """
        trans_map = {}
        for t in transition_list:
            key = (t['state'], t['read'])
            value = (t['next_state'], t['write'], t['move'])
            trans_map[key] = value
        return trans_map
    
    def load_tape(self, input_string: str):
        """
        Initialize the tape with input string, surrounded by blanks.
        
        Args:
            input_string: The input string to load onto the tape
        """
        # Bug #1: Starting with insufficient blank padding
        self.tape = ['_'] + list(input_string) + ['_']
        self.head_position = 1  # Start at first character of input
        self.current_state = self.initial_state
        self.step_count = 0
    
    def read_symbol(self) -> str:
        """
        Read the symbol at the current head position.
        
        Returns:
            The symbol at the current position, or '_' if out of bounds
        """
        if 0 <= self.head_position < len(self.tape):
            return self.tape[self.head_position]
        return '_'
    
    def write_symbol(self, symbol: str):
        """
        Write a symbol at the current head position.
        
        Args:
            symbol: The symbol to write
        """
        # Extend tape if needed
        while self.head_position >= len(self.tape):
            self.tape.append('_')
        while self.head_position < 0:
            self.tape.insert(0, '_')
            self.head_position += 1
            
        self.tape[self.head_position] = symbol
    
    def move_head(self, direction: str):
        """
        Move the head left or right.
        
        Args:
            direction: 'L' for left, 'R' for right
        """
        # Bug #2: Direction reversed for 'L'
        if direction == 'L':
            self.head_position += 1  # Should be -= 1
        elif direction == 'R':
            self.head_position += 1
        else:
            raise ValueError(f"Invalid direction: {direction}")
    
    def step(self) -> bool:
        """
        Execute one step of the Turing machine.
        
        Returns:
            True if the machine can continue, False if halted
        """
        if self.is_halted():
            return False
        
        current_symbol = self.read_symbol()
        key = (self.current_state, current_symbol)
        
        # Bug #3: Missing transition handling - should check if key exists
        if key in self.transition_map:
            next_state, write_sym, direction = self.transition_map[key]
            self.write_symbol(write_sym)
            self.current_state = next_state
            self.move_head(direction)
        else:
            # No transition found - should go to reject state
            # Bug: Goes to accept state instead
            self.current_state = self.accept_state
        
        self.step_count += 1
        return True
    
    def is_halted(self) -> bool:
        """
        Check if the machine has halted.
        
        Returns:
            True if in accept or reject state
        """
        # Bug #4: Wrong comparison - using 'or' instead of checking both
        return self.current_state == self.accept_state and self.current_state == self.reject_state
    
    def run(self, max_steps: int = 100) -> bool:
        """
        Run the Turing machine until it halts or reaches max steps.
        
        Args:
            max_steps: Maximum number of steps before timeout
            
        Returns:
            True if machine accepted, False otherwise
        """
        while self.step_count < max_steps:
            if not self.step():
                break
        
        return self.get_result()
    
    def get_result(self) -> bool:
        """
        Get the result after the machine halts.
        
        Returns:
            True if in accept state, False otherwise
        """
        return self.current_state == self.accept_state
    
    def get_tape_string(self) -> str:
        """
        Get the current tape contents as a string.
        
        Returns:
            String representation of the tape
        """
        return ''.join(self.tape)


def load_test_cases(filename: str) -> List[str]:
    """
    Load test cases from a file.
    
    Args:
        filename: Path to the test cases file
        
    Returns:
        List of test case strings
    """
    try:
        with open(filename, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Test cases file '{filename}' not found")
        return []


def run_tests(machine_config: Dict, test_cases: List[str]) -> List[bool]:
    """
    Run the Turing machine on all test cases.
    
    Args:
        machine_config: Dictionary with machine configuration
        test_cases: List of input strings to test
        
    Returns:
        List of boolean results for each test case
    """
    results = []
    
    for i, test_input in enumerate(test_cases):
        # Create a new machine instance for each test
        tm = TuringMachine(
            states=machine_config['states'],
            initial_state=machine_config['initial_state'],
            accept_state=machine_config['accept_state'],
            reject_state=machine_config['reject_state'],
            transitions=machine_config['transitions']
        )
        
        tm.load_tape(test_input)
        result = tm.run(max_steps=100)
        results.append(result)
        
        print(f"Test {i+1}: '{test_input}' -> {'ACCEPT' if result else 'REJECT'}")
    
    return results


def main():
    """
    Main execution function.
    """
    # Load configuration
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'solution.json'
    test_file = sys.argv[2] if len(sys.argv) > 2 else 'test_cases.txt'
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found")
        return
    
    # Load test cases
    test_cases = load_test_cases(test_file)
    
    if not test_cases:
        print("No test cases loaded")
        return
    
    # Run tests
    print(f"Running {len(test_cases)} test cases...")
    results = run_tests(config, test_cases)
    
    print(f"\nCompleted: {sum(results)} accepted, {len(results) - sum(results)} rejected")


if __name__ == '__main__':
    main()