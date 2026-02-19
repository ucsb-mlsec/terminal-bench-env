#!/usr/bin/env python3

import numpy as np
import json
from scoring_function import evaluate_config

def optimize_parameters(budget=100):
    """
    Optimize parameters using a combination of random search and local refinement.
    """
    
    # Track best parameters and scores
    best_params = None
    best_score = -np.inf
    
    # Keep track of all evaluations
    all_evaluations = []
    
    # Phase 1: Initial random search (40 evaluations)
    print("Phase 1: Random search...")
    random_budget = 40
    
    for i in range(random_budget):
        alpha = np.random.uniform(0.0, 1.0)
        beta = np.random.uniform(0.0, 10.0)
        gamma = np.random.randint(1, 101)
        
        score = evaluate_config(alpha=alpha, beta=beta, gamma=gamma)
        all_evaluations.append((alpha, beta, gamma, score))
        
        if score > best_score:
            best_score = score
            best_params = (alpha, beta, gamma)
            print(f"  New best at eval {i+1}: alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma}, score={score:.4f}")
    
    evaluations_used = random_budget
    
    # Phase 2: Focus on promising regions (30 evaluations)
    print("\nPhase 2: Focused search around best regions...")
    
    # Sort evaluations by score and take top 5
    sorted_evals = sorted(all_evaluations, key=lambda x: x[3], reverse=True)
    top_candidates = sorted_evals[:5]
    
    focused_budget = 30
    for i in range(focused_budget):
        # Choose a random top candidate to explore around
        base_alpha, base_beta, base_gamma, _ = top_candidates[np.random.randint(0, len(top_candidates))]
        
        # Add noise to explore nearby
        alpha = np.clip(base_alpha + np.random.normal(0, 0.1), 0.0, 1.0)
        beta = np.clip(base_beta + np.random.normal(0, 0.5), 0.0, 10.0)
        gamma = int(np.clip(base_gamma + np.random.normal(0, 5), 1, 100))
        
        score = evaluate_config(alpha=alpha, beta=beta, gamma=gamma)
        all_evaluations.append((alpha, beta, gamma, score))
        
        if score > best_score:
            best_score = score
            best_params = (alpha, beta, gamma)
            print(f"  New best at eval {evaluations_used+i+1}: alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma}, score={score:.4f}")
    
    evaluations_used += focused_budget
    
    # Phase 3: Fine-tuning around the best solution (30 evaluations)
    print("\nPhase 3: Fine-tuning...")
    
    fine_tune_budget = budget - evaluations_used
    best_alpha, best_beta, best_gamma = best_params
    
    for i in range(fine_tune_budget):
        # Very small perturbations around best
        alpha = np.clip(best_alpha + np.random.normal(0, 0.05), 0.0, 1.0)
        beta = np.clip(best_beta + np.random.normal(0, 0.3), 0.0, 10.0)
        gamma = int(np.clip(best_gamma + np.random.normal(0, 3), 1, 100))
        
        score = evaluate_config(alpha=alpha, beta=beta, gamma=gamma)
        
        if score > best_score:
            best_score = score
            best_params = (alpha, beta, gamma)
            best_alpha, best_beta, best_gamma = best_params
            print(f"  New best at eval {evaluations_used+i+1}: alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma}, score={score:.4f}")
    
    return best_params

def validate_parameters(alpha, beta, gamma):
    """
    Validate parameters are within valid ranges.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha}")
    if not (0.0 <= beta <= 10.0):
        raise ValueError(f"beta must be between 0.0 and 10.0, got {beta}")
    if not (1 <= gamma <= 100):
        raise ValueError(f"gamma must be between 1 and 100, got {gamma}")
    if not isinstance(gamma, int):
        raise ValueError(f"gamma must be an integer, got {type(gamma)}")

def evaluate_average_score(alpha, beta, gamma, n_evals=10):
    """
    Evaluate average score over multiple runs to account for noise.
    """
    scores = []
    for _ in range(n_evals):
        score = evaluate_config(alpha=alpha, beta=beta, gamma=gamma)
        scores.append(score)
    return np.mean(scores), np.std(scores)

def main():
    print("Starting parameter optimization...")
    print(f"Baseline (alpha=0.5, beta=5.0, gamma=50):")
    baseline_mean, baseline_std = evaluate_average_score(0.5, 5.0, 50, n_evals=5)
    print(f"  Average score: {baseline_mean:.4f} (±{baseline_std:.4f})\n")
    
    # Run optimization
    best_alpha, best_beta, best_gamma = optimize_parameters(budget=100)
    
    print("\n" + "="*60)
    print("Optimization complete!")
    print(f"Best parameters found:")
    print(f"  alpha: {best_alpha:.6f}")
    print(f"  beta: {best_beta:.6f}")
    print(f"  gamma: {best_gamma}")
    
    # Validate and evaluate final parameters
    validate_parameters(best_alpha, best_beta, best_gamma)
    
    print("\nEvaluating final parameters (10 runs)...")
    final_mean, final_std = evaluate_average_score(best_alpha, best_beta, best_gamma, n_evals=10)
    print(f"  Average score: {final_mean:.4f} (±{final_std:.4f})")
    
    if final_mean >= 0.80:
        print("  ✓ SUCCESS: Achieved target score of 0.80 or higher!")
    else:
        print(f"  ✗ Warning: Score {final_mean:.4f} is below target of 0.80")
    
    # Save optimal parameters to JSON
    optimal_params = {
        "alpha": float(best_alpha),
        "beta": float(best_beta),
        "gamma": int(best_gamma)
    }
    
    output_path = "/workspace/optimal_params.json"
    with open(output_path, 'w') as f:
        json.dump(optimal_params, f, indent=2)
    
    print(f"\nOptimal parameters saved to {output_path}")
    print("="*60)

if __name__ == "__main__":
    main()