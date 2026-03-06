#!/usr/bin/env python3

import threading
import random
from typing import Dict, List, Tuple

class Game:
    def __init__(self, number: int):
        self.number = number
    
    def is_over(self) -> bool:
        return self.number == 0
    
    def get_legal_moves(self) -> List[int]:
        return [i for i in [1, 2, 3] if i <= self.number]
    
    def apply_move(self, move: int) -> 'Game':
        return Game(self.number - move)
    
    def copy(self) -> 'Game':
        return Game(self.number)


def simulate_game(game: Game, rng: random.Random) -> int:
    """
    Simulate a random game from the current position.
    Returns 1 if current player wins, 0 if current player loses.
    """
    current_game = game.copy()
    current_player = 0
    
    while not current_game.is_over():
        legal_moves = current_game.get_legal_moves()
        if not legal_moves:
            # Current player loses
            return 0 if current_player == 0 else 1
        
        move = rng.choice(legal_moves)
        current_game = current_game.apply_move(move)
        current_player = 1 - current_player
    
    # Game is over, current_player who just moved won
    return 1 if current_player == 1 else 0


def worker_search(start_number: int, iterations: int, shared_stats: Dict, 
                  lock: threading.Lock, random_seed: int, thread_id: int):
    """
    Worker thread that performs Monte Carlo searches.
    Critical flaw: No coordination about which positions are being explored.
    """
    # Each thread gets its own RNG seeded differently
    rng = random.Random(random_seed + thread_id)
    
    game = Game(start_number)
    legal_first_moves = game.get_legal_moves()
    
    # Local statistics
    local_stats = {move: {'visits': 0, 'wins': 0} for move in legal_first_moves}
    
    for _ in range(iterations):
        # Choose a random first move to explore
        first_move = rng.choice(legal_first_moves)
        
        # No coordination here - multiple threads can explore same position simultaneously
        # This is the critical flaw!
        
        # Apply first move
        game_after_move = game.apply_move(first_move)
        
        # Simulate rest of game
        result = simulate_game(game_after_move, rng)
        
        # Update local statistics
        local_stats[first_move]['visits'] += 1
        local_stats[first_move]['wins'] += result
    
    # Merge local statistics into shared statistics with lock
    with lock:
        for move, stats in local_stats.items():
            if move not in shared_stats:
                shared_stats[move] = {'visits': 0, 'wins': 0}
            shared_stats[move]['visits'] += stats['visits']
            shared_stats[move]['wins'] += stats['wins']


def search_parallel(start_number: int, total_iterations: int, 
                   num_threads: int = 4, random_seed: int = 42) -> Dict:
    """
    Perform parallel Monte Carlo search across multiple threads.
    
    Flaw: Threads don't coordinate which positions they explore,
    leading to redundant work and inconsistent statistics.
    """
    random.seed(random_seed)
    
    # Shared statistics across all threads
    shared_stats = {}
    lock = threading.Lock()
    
    # Divide iterations among threads
    iterations_per_thread = total_iterations // num_threads
    
    # Create and start worker threads
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(
            target=worker_search,
            args=(start_number, iterations_per_thread, shared_stats, 
                  lock, random_seed, i)
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Calculate win rates
    results = {}
    for move, stats in shared_stats.items():
        win_rate = stats['wins'] / stats['visits'] if stats['visits'] > 0 else 0.0
        results[move] = {
            'visits': stats['visits'],
            'wins': stats['wins'],
            'win_rate': win_rate
        }
    
    return results


def get_best_move(results: Dict) -> Tuple[int, float]:
    """Return the move with highest win rate."""
    best_move = None
    best_win_rate = -1.0
    
    for move, stats in results.items():
        if stats['win_rate'] > best_win_rate:
            best_win_rate = stats['win_rate']
            best_move = move
    
    return best_move, best_win_rate


if __name__ == '__main__':
    # Test parameters
    START_NUMBER = 10
    TOTAL_ITERATIONS = 800
    NUM_THREADS = 4
    RANDOM_SEED = 42
    
    print(f"Running parallel search with {NUM_THREADS} threads...")
    print(f"Start number: {START_NUMBER}")
    print(f"Total iterations: {TOTAL_ITERATIONS}")
    print(f"Random seed: {RANDOM_SEED}")
    print()
    
    results = search_parallel(START_NUMBER, TOTAL_ITERATIONS, NUM_THREADS, RANDOM_SEED)
    
    print("Results by move:")
    for move in sorted(results.keys()):
        stats = results[move]
        print(f"  Move {move}: {stats['visits']} visits, "
              f"{stats['wins']} wins, win_rate={stats['win_rate']:.2f}")
    
    best_move, best_win_rate = get_best_move(results)
    print()
    print(f"Recommended move: {best_move} (win_rate={best_win_rate:.2f})")