#!/usr/bin/env python3
"""
Direct Preference Optimization (DPO) Training Script
Implements DPO loss calculation for language model alignment.
"""

import numpy as np
import json
import sys

def log_sigmoid(x):
    """
    Numerically stable log-sigmoid function.
    log(sigmoid(x)) = log(1 / (1 + exp(-x))) = -log(1 + exp(-x))
    """
    return -np.log1p(np.exp(-x))

def compute_dpo_loss(policy_chosen_logprob, policy_rejected_logprob, 
                     ref_chosen_logprob, ref_rejected_logprob, beta):
    """
    Compute DPO loss for a single preference pair.
    
    The DPO loss is based on the Bradley-Terry model:
    Loss = -log(sigmoid(beta * (log_ratio_policy - log_ratio_ref)))
    
    Where:
    - log_ratio_policy = log(P_policy(chosen) / P_policy(rejected))
    - log_ratio_ref = log(P_ref(chosen) / P_ref(rejected))
    
    Args:
        policy_chosen_logprob: Log probability of chosen response under policy model
        policy_rejected_logprob: Log probability of rejected response under policy model
        ref_chosen_logprob: Log probability of chosen response under reference model
        ref_rejected_logprob: Log probability of rejected response under reference model
        beta: Temperature parameter controlling strength of optimization
    
    Returns:
        DPO loss value (scalar)
    """
    # Compute log probability ratios for policy and reference models
    policy_log_ratio = policy_chosen_logprob - policy_rejected_logprob
    ref_log_ratio = ref_chosen_logprob - ref_rejected_logprob
    
    # BUG: The sign is incorrect here - should be subtraction, not addition
    logits = beta * (policy_log_ratio + ref_log_ratio)
    
    # Apply negative log sigmoid to get loss
    loss = -log_sigmoid(logits)
    
    return loss

def main():
    """
    Main training function that loads test data and computes DPO losses.
    """
    # Load test preferences
    try:
        with open('/workspace/test_preferences.json', 'r') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print("Error: Test preferences file not found")
        sys.exit(1)
    
    print("Running DPO loss computation on test dataset...")
    print("=" * 60)
    
    # Process each test case
    for i, test_case in enumerate(test_data):
        loss = compute_dpo_loss(
            test_case['policy_chosen_logprob'],
            test_case['policy_rejected_logprob'],
            test_case['ref_chosen_logprob'],
            test_case['ref_rejected_logprob'],
            test_case['beta']
        )
        
        expected_loss = test_case['expected_loss']
        error = abs(loss - expected_loss)
        
        print(f"Test case {i+1}:")
        print(f"  Computed loss: {loss:.6f}")
        print(f"  Expected loss: {expected_loss:.6f}")
        print(f"  Error: {error:.6f}")
        print()

if __name__ == "__main__":
    main()