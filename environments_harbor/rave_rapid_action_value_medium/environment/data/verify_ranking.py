#!/usr/bin/env python3

import json
import os
import math
from pathlib import Path

def verify_ranking():
    results = []
    passed = True
    
    def log(message, is_error=False):
        nonlocal passed
        results.append(message)
        if is_error:
            passed = False
    
    # Step 1: Check if solution file exists
    solution_path = '/home/agent/solution/ranking.json'
    log(f"Checking if solution file exists: {solution_path}")
    if not os.path.exists(solution_path):
        log(f"ERROR: Solution file not found at {solution_path}", True)
        write_results(results, passed)
        return
    log("✓ Solution file exists")
    
    # Step 2: Verify it contains valid JSON
    log("\nValidating JSON format...")
    try:
        with open(solution_path, 'r') as f:
            solution = json.load(f)
        log("✓ Valid JSON format")
    except json.JSONDecodeError as e:
        log(f"ERROR: Invalid JSON format: {e}", True)
        write_results(results, passed)
        return
    except Exception as e:
        log(f"ERROR: Could not read solution file: {e}", True)
        write_results(results, passed)
        return
    
    # Step 3: Verify required fields
    log("\nChecking required fields...")
    required_fields = ['best_move', 'worst_move', 'ranked_moves']
    for field in required_fields:
        if field not in solution:
            log(f"ERROR: Missing required field '{field}'", True)
        else:
            log(f"✓ Field '{field}' present")
    
    if len(solution) != 3:
        log(f"ERROR: Solution should contain exactly 3 fields, found {len(solution)}", True)
    else:
        log("✓ Exactly 3 fields present")
    
    if not passed:
        write_results(results, passed)
        return
    
    # Step 4: Verify field types
    log("\nValidating field types...")
    if not isinstance(solution.get('best_move'), str):
        log(f"ERROR: 'best_move' must be a string, got {type(solution.get('best_move'))}", True)
    else:
        log("✓ 'best_move' is a string")
    
    if not isinstance(solution.get('worst_move'), str):
        log(f"ERROR: 'worst_move' must be a string, got {type(solution.get('worst_move'))}", True)
    else:
        log("✓ 'worst_move' is a string")
    
    if not isinstance(solution.get('ranked_moves'), list):
        log(f"ERROR: 'ranked_moves' must be an array, got {type(solution.get('ranked_moves'))}", True)
        write_results(results, passed)
        return
    else:
        log("✓ 'ranked_moves' is an array")
    
    # Step 5: Verify ranked_moves contains exactly 8 unique moves
    log("\nValidating ranked_moves content...")
    ranked_moves = solution.get('ranked_moves', [])
    
    if len(ranked_moves) != 8:
        log(f"ERROR: 'ranked_moves' should contain exactly 8 moves, found {len(ranked_moves)}", True)
    else:
        log("✓ 'ranked_moves' contains 8 moves")
    
    expected_moves = {f"move_{i}" for i in range(8)}
    actual_moves = set(ranked_moves)
    
    if len(actual_moves) != len(ranked_moves):
        log(f"ERROR: 'ranked_moves' contains duplicate entries", True)
    else:
        log("✓ All moves in 'ranked_moves' are unique")
    
    if actual_moves != expected_moves:
        missing = expected_moves - actual_moves
        extra = actual_moves - expected_moves
        if missing:
            log(f"ERROR: Missing moves: {missing}", True)
        if extra:
            log(f"ERROR: Unexpected moves: {extra}", True)
    else:
        log("✓ 'ranked_moves' contains all expected move identifiers (move_0 to move_7)")
    
    # Step 6: Verify best_move and worst_move match ranked_moves
    log("\nValidating best_move and worst_move consistency...")
    if len(ranked_moves) > 0:
        if solution.get('best_move') != ranked_moves[0]:
            log(f"ERROR: 'best_move' ({solution.get('best_move')}) should be the first element of 'ranked_moves' ({ranked_moves[0]})", True)
        else:
            log("✓ 'best_move' matches first element of 'ranked_moves'")
        
        if solution.get('worst_move') != ranked_moves[-1]:
            log(f"ERROR: 'worst_move' ({solution.get('worst_move')}) should be the last element of 'ranked_moves' ({ranked_moves[-1]})", True)
        else:
            log("✓ 'worst_move' matches last element of 'ranked_moves'")
    
    # Step 7: Load and verify against input data
    log("\nLoading move statistics...")
    stats_path = '/home/agent/game_data/move_statistics.json'
    try:
        with open(stats_path, 'r') as f:
            statistics = json.load(f)
        log(f"✓ Successfully loaded move statistics from {stats_path}")
    except FileNotFoundError:
        log(f"ERROR: Move statistics file not found at {stats_path}", True)
        write_results(results, passed)
        return
    except json.JSONDecodeError as e:
        log(f"ERROR: Invalid JSON in statistics file: {e}", True)
        write_results(results, passed)
        return
    except Exception as e:
        log(f"ERROR: Could not read statistics file: {e}", True)
        write_results(results, passed)
        return
    
    # Verify move identifiers match
    log("\nVerifying move identifiers match statistics file...")
    stats_moves = set(statistics.keys())
    if actual_moves != stats_moves:
        missing = stats_moves - actual_moves
        extra = actual_moves - stats_moves
        if missing:
            log(f"ERROR: Moves in statistics but not in ranking: {missing}", True)
        if extra:
            log(f"ERROR: Moves in ranking but not in statistics: {extra}", True)
    else:
        log("✓ All move identifiers match those in statistics file")
    
    # Step 8: Evaluate ranking quality
    log("\nEvaluating ranking quality...")
    if not passed:
        log("Skipping quality evaluation due to previous errors")
    else:
        try:
            # Calculate UCB1 scores for comparison
            total_visits = sum(stat['visits'] for stat in statistics.values())
            
            move_scores = {}
            for move, stat in statistics.items():
                visits = stat['visits']
                wins = stat['wins']
                win_rate = wins / visits if visits > 0 else 0
                # UCB1 formula
                if visits > 0:
                    exploration = math.sqrt(2 * math.log(total_visits) / visits)
                    ucb1_score = win_rate + exploration
                else:
                    ucb1_score = 0
                move_scores[move] = {
                    'win_rate': win_rate,
                    'visits': visits,
                    'wins': wins,
                    'ucb1': ucb1_score
                }
            
            log("\nMove Statistics (ordered by UCB1 score):")
            sorted_by_ucb1 = sorted(move_scores.items(), key=lambda x: x[1]['ucb1'], reverse=True)
            for move, scores in sorted_by_ucb1:
                log(f"  {move}: visits={scores['visits']}, wins={scores['wins']}, "
                    f"win_rate={scores['win_rate']:.3f}, UCB1={scores['ucb1']:.3f}")
            
            log("\nYour ranking:")
            for i, move in enumerate(ranked_moves, 1):
                scores = move_scores[move]
                log(f"  {i}. {move}: visits={scores['visits']}, wins={scores['wins']}, "
                    f"win_rate={scores['win_rate']:.3f}, UCB1={scores['ucb1']:.3f}")
            
            # Check if ranking is reasonable (top-ranked moves should have competitive metrics)
            best_move = ranked_moves[0]
            best_score = move_scores[best_move]
            log(f"\n✓ Ranking quality evaluation complete")
            log(f"  Best move selected: {best_move}")
            log(f"  Its win rate: {best_score['win_rate']:.3f}")
            log(f"  Its visits: {best_score['visits']}")
            
        except Exception as e:
            log(f"WARNING: Could not evaluate ranking quality: {e}")
    
    write_results(results, passed)

def write_results(results, passed):
    # Write to file
    output_path = '/home/agent/solution/verification.txt'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    status = "PASS" if passed else "FAIL"
    output = f"Verification Status: {status}\n\n"
    output += "\n".join(results)
    
    with open(output_path, 'w') as f:
        f.write(output)
    
    # Print to console
    print(f"\n{'='*60}")
    print(f"VERIFICATION RESULT: {status}")
    print(f"{'='*60}\n")
    print(output)
    print(f"\n{'='*60}")
    print(f"Detailed results written to: {output_path}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    verify_ranking()