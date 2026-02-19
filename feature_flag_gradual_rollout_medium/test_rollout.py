#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '/app/service')

from feature_flags import FeatureFlagService

def test_feature_flags():
    # Load the service
    config_path = '/app/config/features.json'
    service = FeatureFlagService(config_path)
    
    # Test with a large sample of user IDs
    num_users = 10000
    user_ids = list(range(1, num_users + 1))
    
    # Get all features from config
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    features = config.get('features', [])
    
    all_tests_passed = True
    
    print("=" * 80)
    print("FEATURE FLAG ROLLOUT TESTS")
    print("=" * 80)
    print(f"Testing with {num_users} user IDs\n")
    
    for feature in features:
        feature_name = feature['name']
        expected_percentage = feature['rollout_percentage']
        
        # Count how many users have the feature enabled
        enabled_count = 0
        for user_id in user_ids:
            if service.is_feature_enabled(feature_name, user_id):
                enabled_count += 1
        
        actual_percentage = (enabled_count / num_users) * 100
        
        # Check if within ±5% tolerance
        tolerance = 5.0
        within_tolerance = abs(actual_percentage - expected_percentage) <= tolerance
        
        # Test determinism - same user should always get same result
        test_user_id = 12345
        result1 = service.is_feature_enabled(feature_name, test_user_id)
        result2 = service.is_feature_enabled(feature_name, test_user_id)
        deterministic = (result1 == result2)
        
        # Test distribution - check that we get both True and False results
        # (unless rollout is 0% or 100%)
        results_set = set()
        for user_id in user_ids[:100]:
            results_set.add(service.is_feature_enabled(feature_name, user_id))
        
        has_distribution = True
        if 0 < expected_percentage < 100:
            has_distribution = len(results_set) > 1
        
        # Overall test result
        test_passed = within_tolerance and deterministic and has_distribution
        
        # Print results
        status = "✓ PASS" if test_passed else "✗ FAIL"
        print(f"Feature: {feature_name}")
        print(f"  Expected: {expected_percentage}%")
        print(f"  Actual: {actual_percentage:.2f}%")
        print(f"  Difference: {abs(actual_percentage - expected_percentage):.2f}%")
        print(f"  Within tolerance (±{tolerance}%): {within_tolerance}")
        print(f"  Deterministic: {deterministic}")
        print(f"  Distribution check: {has_distribution}")
        print(f"  Status: {status}")
        print()
        
        if not test_passed:
            all_tests_passed = False
    
    print("=" * 80)
    
    if all_tests_passed:
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        sys.exit(0)
    else:
        print("SOME TESTS FAILED ✗")
        print("=" * 80)
        sys.exit(1)

if __name__ == '__main__':
    test_feature_flags()