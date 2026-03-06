#!/usr/bin/env python3

import json
import numpy as np

# Reference implementation functions

def cubic_bezier_point(control_points, u):
    """
    Compute a point on a cubic Bezier curve at parameter u.
    
    Args:
        control_points: List of 4 3D points (each is a list/tuple of 3 floats)
        u: Parameter value in [0, 1]
    
    Returns:
        3D point on the curve as numpy array
    """
    P0, P1, P2, P3 = [np.array(p) for p in control_points]
    
    # Cubic Bezier formula: B(u) = (1-u)³P0 + 3(1-u)²u*P1 + 3(1-u)u²P2 + u³P3
    b0 = (1 - u) ** 3
    b1 = 3 * (1 - u) ** 2 * u
    b2 = 3 * (1 - u) * u ** 2
    b3 = u ** 3
    
    return b0 * P0 + b1 * P1 + b2 * P2 + b3 * P3


def find_ray_curve_intersection(ray_origin, ray_direction, control_points):
    """
    Find the intersection between a ray and a cubic Bezier curve.
    
    Args:
        ray_origin: 3D point as list of 3 floats
        ray_direction: Normalized 3D vector as list of 3 floats
        control_points: List of 4 3D points defining the cubic Bezier curve
    
    Returns:
        The smallest non-negative t value where intersection occurs, or None
    """
    origin = np.array(ray_origin)
    direction = np.array(ray_direction)
    
    # Normalize direction to be safe
    direction = direction / np.linalg.norm(direction)
    
    # Sample the curve at many points to find intersections
    num_samples = 500
    tolerance = 0.00001  # Distance tolerance for considering a point on the ray
    
    valid_t_values = []
    
    for i in range(num_samples + 1):
        u = i / num_samples
        
        # Get point on curve at parameter u
        curve_point = cubic_bezier_point(control_points, u)
        
        # Compute the ray parameter t that gives the closest point on the ray to curve_point
        # The closest point on ray to curve_point is: origin + t * direction
        # where t = dot(curve_point - origin, direction) / dot(direction, direction)
        diff = curve_point - origin
        t = np.dot(diff, direction) / np.dot(direction, direction)
        
        # Check if t is valid (non-negative)
        if t < 0:
            continue
        
        # Compute the actual closest point on the ray
        ray_point = origin + t * direction
        
        # Check if the curve point is close enough to the ray
        distance = np.linalg.norm(curve_point - ray_point)
        
        if distance < tolerance:
            valid_t_values.append(t)
    
    # Return the minimum valid t value, or None if no intersections found
    if valid_t_values:
        return min(valid_t_values)
    else:
        return None


# Main validation logic

def load_test_cases(filename):
    """Load test cases from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def is_result_incorrect(test_case, correct_t):
    """
    Check if a test case's claimed result is incorrect.
    
    Args:
        test_case: Dictionary containing the test case data
        correct_t: The correct t value (float or None)
    
    Returns:
        True if the claimed result is incorrect, False otherwise
    """
    claimed_t = test_case.get('t')
    
    # Case 1: Claimed intersection but no actual intersection
    if claimed_t is not None and correct_t is None:
        return True
    
    # Case 2: Claimed no intersection but actual intersection exists
    if claimed_t is None and correct_t is not None:
        return True
    
    # Case 3: Both claim intersection, check if t values match within tolerance
    if claimed_t is not None and correct_t is not None:
        if abs(claimed_t - correct_t) > 0.001:
            return True
    
    # Both None or t values match within tolerance
    return False


def main():
    # Load test cases
    test_cases = load_test_cases('test_cases.json')
    
    invalid_case_ids = []
    
    # Process each test case
    for test_case in test_cases:
        case_id = test_case['id']
        ray_origin = test_case['ray_origin']
        ray_direction = test_case['ray_direction']
        control_points = test_case['control_points']
        
        # Compute correct intersection using reference implementation
        correct_t = find_ray_curve_intersection(ray_origin, ray_direction, control_points)
        
        # Check if the claimed result is incorrect
        if is_result_incorrect(test_case, correct_t):
            invalid_case_ids.append(case_id)
    
    # Sort the invalid case IDs alphabetically
    invalid_case_ids.sort()
    
    # Write results to file
    with open('/workspace/invalid_cases.txt', 'w') as f:
        for case_id in invalid_case_ids:
            f.write(case_id + '\n')


if __name__ == '__main__':
    main()