import time
import math

def evaluate_config(alpha, beta, gamma):
    """
    Evaluate configuration parameters and return a performance score.
    
    Args:
        alpha: float between 0.001 and 1.0
        beta: float between 0.1 and 10.0
        gamma: integer between 1 and 100
    
    Returns:
        float: Performance score (higher is better)
    """
    # Validate parameters
    if not (0.001 <= alpha <= 1.0):
        raise ValueError(f"alpha must be between 0.001 and 1.0, got {alpha}")
    if not (0.1 <= beta <= 10.0):
        raise ValueError(f"beta must be between 0.1 and 10.0, got {beta}")
    if not (1 <= gamma <= 100):
        raise ValueError(f"gamma must be between 1 and 100, got {gamma}")
    if not isinstance(gamma, int):
        raise ValueError(f"gamma must be an integer, got {type(gamma)}")
    
    # Simulate expensive computation
    time.sleep(0.2)
    
    # Complex non-linear scoring function with multiple local maxima
    # Designed to have global maximum around alpha=0.15, beta=2.5, gamma=33
    
    # Normalize parameters to similar scales
    alpha_norm = alpha
    beta_norm = beta / 10.0
    gamma_norm = gamma / 100.0
    
    # Component 1: Trigonometric interaction
    trig_component = math.sin(alpha_norm * 15 + 1.5) * math.cos(beta_norm * 20 - 2.0)
    
    # Component 2: Exponential decay from optimal points
    alpha_optimal = 0.15
    beta_optimal = 2.5
    gamma_optimal = 33
    
    distance_alpha = abs(alpha - alpha_optimal)
    distance_beta = abs(beta - beta_optimal)
    distance_gamma = abs(gamma - gamma_optimal)
    
    exp_component = math.exp(-3 * distance_alpha) * math.exp(-0.3 * distance_beta) * math.exp(-0.02 * distance_gamma)
    
    # Component 3: Polynomial with multiple peaks
    poly_component = (
        0.3 * (1 - (alpha_norm - 0.15) ** 2) +
        0.2 * (1 - (beta_norm - 0.25) ** 2) +
        0.15 * math.sin(gamma_norm * 10)
    )
    
    # Component 4: Interaction terms
    interaction = 0.1 * math.sin(alpha * beta) * math.cos(gamma / 10.0)
    
    # Component 5: Secondary peak to create local maxima
    secondary_peak = 0.15 * math.exp(-((alpha - 0.6) ** 2 + (beta - 7.0) ** 2 + (gamma - 75) ** 2) / 50)
    
    # Combine all components
    raw_score = (
        0.4 * exp_component +
        0.25 * trig_component +
        0.2 * poly_component +
        0.1 * interaction +
        secondary_peak
    )
    
    # Scale and shift to get desired range
    # This should give scores mostly between 0.3-0.7 for random values
    # and above 0.85 for near-optimal values
    score = 0.5 + 0.5 * raw_score
    
    # Ensure score is in valid range
    score = max(0.0, min(1.0, score))
    
    return score


if __name__ == "__main__":
    # Test the function with some sample values
    print("Testing scoring function:")
    print(f"Random test 1: alpha=0.5, beta=5.0, gamma=50")
    score1 = evaluate_config(0.5, 5.0, 50)
    print(f"Score: {score1:.3f}\n")
    
    print(f"Random test 2: alpha=0.1, beta=1.0, gamma=10")
    score2 = evaluate_config(0.1, 1.0, 10)
    print(f"Score: {score2:.3f}\n")
    
    print(f"Near-optimal test: alpha=0.15, beta=2.5, gamma=33")
    score3 = evaluate_config(0.15, 2.5, 33)
    print(f"Score: {score3:.3f}\n")
    
    print(f"Boundary test: alpha=0.001, beta=0.1, gamma=1")
    score4 = evaluate_config(0.001, 0.1, 1)
    print(f"Score: {score4:.3f}")