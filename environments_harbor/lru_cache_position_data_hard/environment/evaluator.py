#!/usr/bin/env python3
"""
DO NOT MODIFY - This simulates an expensive evaluation function

This module contains a chess position evaluation function that is intentionally
slow to simulate expensive computation during game analysis.
"""

import time
import hashlib


def evaluate_position(position: str) -> float:
    """
    Evaluate a chess position and return a score.
    
    This function simulates an expensive computational process by introducing
    a deliberate delay. In a real chess engine, this would involve complex
    position analysis, pattern recognition, and strategic evaluation.
    
    Args:
        position: A chess position in FEN notation
        
    Returns:
        A floating-point evaluation score between -100.0 and 100.0
        Positive values favor white, negative values favor black
        
    Note:
        This function is intentionally slow (3ms per call) to represent
        a computational bottleneck that would benefit from caching.
    """
    # Simulate expensive computation - DO NOT REMOVE
    # In a real engine, this would be actual position analysis
    time.sleep(0.003)
    
    # Generate a deterministic score based on the position string
    # Same position always returns the same evaluation
    position_hash = hashlib.sha256(position.encode()).hexdigest()
    
    # Convert hash to a numeric value
    hash_int = int(position_hash[:16], 16)
    
    # Scale to range [-100.0, 100.0]
    evaluation = ((hash_int % 20001) - 10000) / 100.0
    
    return evaluation


if __name__ == "__main__":
    # Demonstration of the evaluation function
    sample_position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    print(f"Evaluating position: {sample_position}")
    start = time.time()
    score = evaluate_position(sample_position)
    elapsed = time.time() - start
    
    print(f"Evaluation: {score:.2f}")
    print(f"Time taken: {elapsed*1000:.2f}ms")