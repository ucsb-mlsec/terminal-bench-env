#!/usr/bin/env python3

import json
import math

def dot(v1, v2):
    """Compute dot product of two 3D vectors"""
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]

def cross(v1, v2):
    """Compute cross product of two 3D vectors"""
    return [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
    ]

def sub(v1, v2):
    """Subtract two 3D vectors"""
    return [v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]]

def ray_triangle_intersect(ray_origin, ray_direction, v0, v1, v2):
    """
    Möller-Trumbore ray-triangle intersection algorithm
    
    Args:
        ray_origin: 3D point as list/tuple [x, y, z]
        ray_direction: 3D vector as list/tuple [x, y, z]
        v0, v1, v2: Triangle vertices as 3D points
    
    Returns:
        Dictionary with:
            'hit': boolean indicating if intersection occurred
            'distance': float distance along ray to intersection (or None)
            'barycentric': tuple (u, v, w) barycentric coordinates (or None)
    """
    EPSILON = 1e-8
    
    # Compute triangle edges
    edge1 = sub(v1, v0)
    edge2 = sub(v2, v0)
    
    # Begin calculating determinant - also used to calculate u parameter
    h = cross(ray_direction, edge2)
    a = dot(edge1, h)
    
    # BUG 1: Missing epsilon check - if ray is parallel to triangle, a will be near zero
    # Should check: if abs(a) < EPSILON: return no hit
    if a > -EPSILON and a < EPSILON:
        return {'hit': False, 'distance': None, 'barycentric': None}
    
    f = 1.0 / a
    s = sub(ray_origin, v0)
    u = f * dot(s, h)
    
    # BUG 2: Wrong boundary condition - should be u < 0 or u > 1
    # Current code uses <= which incorrectly rejects edge hits
    if u < 0.0 or u > 1.0:
        return {'hit': False, 'distance': None, 'barycentric': None}
    
    q = cross(s, edge1)
    v = f * dot(ray_direction, q)
    
    # BUG 3: Incorrect barycentric coordinate check
    # Should be: v < 0 or u + v > 1
    # Current has wrong condition
    if v < 0.0 or u + v > 1.0:
        return {'hit': False, 'distance': None, 'barycentric': None}
    
    # Calculate t (distance along ray)
    t = f * dot(edge2, q)
    
    # BUG 4: Missing check for negative t (intersection behind ray origin)
    # Should check: if t < EPSILON: return no hit
    if t > EPSILON:
        # Ray intersection
        w = 1.0 - u - v
        return {
            'hit': True,
            'distance': t,
            'barycentric': (u, v, w)
        }
    else:
        # Line intersection but not ray intersection (behind origin)
        return {'hit': False, 'distance': None, 'barycentric': None}

def run_tests():
    """Run all test cases and return results"""
    with open('/workspace/test_cases.json', 'r') as f:
        test_cases = json.load(f)
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases):
        ray_origin = test['ray']['origin']
        ray_direction = test['ray']['direction']
        v0 = test['triangle']['v0']
        v1 = test['triangle']['v1']
        v2 = test['triangle']['v2']
        expected = test['expected']
        
        result = ray_triangle_intersect(ray_origin, ray_direction, v0, v1, v2)
        
        # Check if hit matches expected
        if result['hit'] != expected['hit']:
            print(f"Test {i} FAILED: hit={result['hit']}, expected={expected['hit']}")
            failed += 1
            continue
        
        # If hit is expected, check distance
        if expected['hit']:
            if result['distance'] is None:
                print(f"Test {i} FAILED: distance is None but hit was True")
                failed += 1
                continue
            
            if abs(result['distance'] - expected['distance']) > 0.0001:
                print(f"Test {i} FAILED: distance={result['distance']}, expected={expected['distance']}")
                failed += 1
                continue
        
        passed += 1
    
    print(f"\nTests passed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)

if __name__ == "__main__":
    all_passed = run_tests()
    
    with open('/workspace/result.txt', 'w') as f:
        f.write(f"bugs_fixed=4\n")
        f.write(f"all_tests_passed={'yes' if all_passed else 'no'}\n")