#!/usr/bin/env python3

import sys
import json
import time
import random
from pathlib import Path

# Add paths to import modules
sys.path.insert(0, '/game')
sys.path.insert(0, '/solution')

try:
    from tictactoe import TicTacToe
except ImportError:
    print("Error: Could not import TicTacToe from /game/tictactoe.py")
    sys.exit(1)

try:
    from ai_player import get_best_move
except ImportError:
    print("Error: Could not import get_best_move from /solution/ai_player.py")
    sys.exit(1)


def play_game(ai_player_num):
    """
    Simulate a complete game with AI as player ai_player_num (1 or 2).
    Opponent plays random valid moves.
    Returns 'win', 'loss', or 'draw' from AI perspective.
    Also returns list of times for each AI move.
    """
    game = TicTacToe()
    current_player = 1
    move_times = []
    
    while not game.is_game_over():
        if current_player == ai_player_num:
            # AI's turn
            start_time = time.time()
            try:
                move = get_best_move(game.get_board(), ai_player_num)
                elapsed = time.time() - start_time
                move_times.append(elapsed)
                
                if not game.make_move(move[0], move[1], current_player):
                    # Invalid move by AI
                    return 'loss', move_times
            except Exception as e:
                print(f"Error in AI move: {e}")
                return 'loss', move_times
        else:
            # Random opponent's turn
            valid_moves = []
            board = game.get_board()
            for i in range(3):
                for j in range(3):
                    if board[i][j] == 0:
                        valid_moves.append((i, j))
            
            if valid_moves:
                move = random.choice(valid_moves)
                game.make_move(move[0], move[1], current_player)
        
        current_player = 3 - current_player  # Switch between 1 and 2
    
    winner = game.get_winner()
    if winner == ai_player_num:
        return 'win', move_times
    elif winner == 0:
        return 'draw', move_times
    else:
        return 'loss', move_times


def test_performance(num_games=100):
    """
    Run num_games against random player.
    AI plays as X (player 1) for half the games.
    AI plays as O (player 2) for other half.
    Returns dictionary with 'win_rate' and 'avg_time_per_move'.
    """
    wins = 0
    draws = 0
    losses = 0
    all_move_times = []
    
    for i in range(num_games):
        ai_player = 1 if i < num_games // 2 else 2
        result, move_times = play_game(ai_player)
        all_move_times.extend(move_times)
        
        if result == 'win':
            wins += 1
        elif result == 'draw':
            draws += 1
        else:
            losses += 1
    
    win_rate = (wins + draws) / num_games
    avg_time_per_move = sum(all_move_times) / len(all_move_times) if all_move_times else 0
    
    print(f"Performance Test Results:")
    print(f"  Wins: {wins}, Draws: {draws}, Losses: {losses}")
    print(f"  Win Rate (including draws): {win_rate:.2%}")
    print(f"  Average time per move: {avg_time_per_move:.4f}s")
    
    return {
        'win_rate': win_rate,
        'avg_time_per_move': avg_time_per_move
    }


def test_standard_positions():
    """
    Test specific board positions to verify AI makes correct moves.
    Returns True if all tests pass.
    """
    print("\nTesting standard positions...")
    
    # Test 1: AI can win in one move (horizontal)
    board1 = [
        [1, 1, 0],
        [2, 2, 0],
        [0, 0, 0]
    ]
    move1 = get_best_move(board1, 1)
    if move1 != (0, 2):
        print(f"  Test 1 FAILED: Expected (0, 2), got {move1}")
        return False
    print("  Test 1 PASSED: AI wins in one move")
    
    # Test 2: AI must block opponent win
    board2 = [
        [2, 2, 0],
        [1, 0, 0],
        [0, 0, 0]
    ]
    move2 = get_best_move(board2, 1)
    if move2 != (0, 2):
        print(f"  Test 2 FAILED: Expected (0, 2), got {move2}")
        return False
    print("  Test 2 PASSED: AI blocks opponent win")
    
    # Test 3: AI can win in one move (diagonal)
    board3 = [
        [1, 2, 0],
        [2, 1, 0],
        [0, 0, 0]
    ]
    move3 = get_best_move(board3, 1)
    if move3 != (2, 2):
        print(f"  Test 3 FAILED: Expected (2, 2), got {move3}")
        return False
    print("  Test 3 PASSED: AI wins with diagonal")
    
    # Test 4: AI must block opponent diagonal win
    board4 = [
        [2, 1, 0],
        [1, 2, 0],
        [0, 0, 0]
    ]
    move4 = get_best_move(board4, 1)
    if move4 != (2, 2):
        print(f"  Test 4 FAILED: Expected (2, 2), got {move4}")
        return False
    print("  Test 4 PASSED: AI blocks diagonal win")
    
    # Test 5: Center move on empty board (common strategy)
    board5 = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    move5 = get_best_move(board5, 1)
    # Center or corner is acceptable on empty board
    valid_moves = [(1, 1), (0, 0), (0, 2), (2, 0), (2, 2)]
    if move5 not in valid_moves:
        print(f"  Test 5 FAILED: Expected center or corner, got {move5}")
        return False
    print("  Test 5 PASSED: AI makes strategic opening move")
    
    print("All standard position tests PASSED")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("TIC-TAC-TOE AI PLAYER EVALUATION")
    print("=" * 60)
    
    # Run performance test
    perf_results = test_performance(100)
    
    # Run standard position tests
    positions_passed = test_standard_positions()
    
    # Determine overall pass/fail
    test_passed = (
        perf_results['win_rate'] >= 0.90 and
        perf_results['avg_time_per_move'] < 2.0 and
        positions_passed
    )
    
    # Create results dictionary
    results = {
        'win_rate': perf_results['win_rate'],
        'avg_time_per_move': perf_results['avg_time_per_move'],
        'test_passed': test_passed
    }
    
    # Write results to JSON file
    output_path = Path('/solution/results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Win Rate: {results['win_rate']:.2%} (requirement: >= 90%)")
    print(f"Avg Time per Move: {results['avg_time_per_move']:.4f}s (requirement: < 2.0s)")
    print(f"Standard Positions: {'PASSED' if positions_passed else 'FAILED'}")
    print(f"\nOverall Test: {'PASSED ✓' if test_passed else 'FAILED ✗'}")
    print("=" * 60)
    print(f"\nResults saved to: {output_path}")
    
    sys.exit(0 if test_passed else 1)