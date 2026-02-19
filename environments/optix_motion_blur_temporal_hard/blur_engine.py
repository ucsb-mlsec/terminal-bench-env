#!/usr/bin/env python3

import json
import math
import sys

def load_animation_data(filename):
    """Load animation data from JSON file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: Animation data file '{filename}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in '{filename}'.")
        sys.exit(1)

def calculate_motion_blur(obj, shutter_interval, num_samples=10):
    """
    Calculate motion blur for a 2D animated object.
    
    Parameters:
        obj: Dictionary with 'position' (dict with 'x', 'y') and 'velocity' (dict with 'vx', 'vy')
        shutter_interval: Float representing exposure duration
        num_samples: Number of temporal samples to take
    
    Returns:
        List of dictionaries containing blur trail coordinates
    """
    pos_x = obj['position']['x']
    pos_y = obj['position']['y']
    vel_x = obj['velocity']['vx']
    vel_y = obj['velocity']['vy']
    
    blur_trail = []
    total_weight = 0.0
    
    # BUG 1: Temporal sampling range is wrong - should be centered around current time
    # Currently goes from 0 to 2*shutter_interval instead of -shutter_interval/2 to +shutter_interval/2
    for i in range(num_samples):
        # Wrong range: 0 to 2*shutter_interval
        t_offset = (i / (num_samples - 1)) * 2.0 * shutter_interval
        
        # BUG 2: Coordinate axis confusion - using vel_y for x and vel_x for y
        sample_x = pos_x + vel_y * t_offset  # Should be vel_x
        sample_y = pos_y + vel_x * t_offset  # Should be vel_y
        
        # BUG 3: Weight calculation uses wrong formula (should be uniform or Gaussian)
        # This creates uneven weighting that doesn't properly represent exposure
        weight = 1.0 + (i / num_samples)  # Should be uniform (1.0) or proper Gaussian
        
        total_weight += weight
        
        blur_trail.append({
            'x': sample_x,
            'y': sample_y,
            'weight': weight
        })
    
    # BUG 4: Missing weight normalization
    # Weights should be normalized to sum to 1.0, but this is commented out or missing
    # for sample in blur_trail:
    #     sample['weight'] /= total_weight
    
    return blur_trail

def compute_blur_metrics(blur_trail, obj):
    """Compute metrics to validate blur characteristics."""
    if len(blur_trail) < 2:
        return {
            'length': 0.0,
            'angle': 0.0,
            'valid': False
        }
    
    # Calculate blur trail length
    start = blur_trail[0]
    end = blur_trail[-1]
    length = math.sqrt((end['x'] - start['x'])**2 + (end['y'] - start['y'])**2)
    
    # Calculate blur angle
    dx = end['x'] - start['x']
    dy = end['y'] - start['y']
    angle = math.degrees(math.atan2(dy, dx))
    
    # Calculate expected angle from velocity
    vel_x = obj['velocity']['vx']
    vel_y = obj['velocity']['vy']
    expected_angle = math.degrees(math.atan2(vel_y, vel_x))
    
    # Calculate velocity magnitude
    velocity_magnitude = math.sqrt(vel_x**2 + vel_y**2)
    
    return {
        'length': length,
        'angle': angle,
        'expected_angle': expected_angle,
        'angle_error': abs(angle - expected_angle),
        'velocity_magnitude': velocity_magnitude,
        'valid': True
    }

def main():
    """Main execution function."""
    print("Motion Blur Engine - Processing animation data...")
    print("=" * 60)
    
    # Load animation data
    data = load_animation_data('/workspace/animation_data.json')
    
    shutter_interval = data.get('shutter_interval', 0.1)
    objects = data.get('objects', [])
    
    print(f"\nShutter interval: {shutter_interval}")
    print(f"Number of objects: {len(objects)}")
    print("\nProcessing objects:\n")
    
    all_valid = True
    
    for idx, obj in enumerate(objects):
        obj_name = obj.get('name', f'Object_{idx}')
        print(f"Object: {obj_name}")
        print(f"  Position: ({obj['position']['x']}, {obj['position']['y']})")
        print(f"  Velocity: ({obj['velocity']['vx']}, {obj['velocity']['vy']})")
        
        # Calculate motion blur
        blur_trail = calculate_motion_blur(obj, shutter_interval, num_samples=10)
        
        # Compute metrics
        metrics = compute_blur_metrics(blur_trail, obj)
        
        if metrics['valid']:
            print(f"  Blur length: {metrics['length']:.2f}")
            print(f"  Blur angle: {metrics['angle']:.2f}°")
            print(f"  Expected angle: {metrics['expected_angle']:.2f}°")
            print(f"  Angle error: {metrics['angle_error']:.2f}°")
            print(f"  Velocity magnitude: {metrics['velocity_magnitude']:.2f}")
            
            # Validation checks
            if metrics['angle_error'] > 5.0:  # More than 5 degrees off
                print(f"  WARNING: Blur angle doesn't match motion direction!")
                all_valid = False
            
            if metrics['velocity_magnitude'] > 0.1 and metrics['length'] < 0.01:
                print(f"  WARNING: Object has velocity but no visible blur!")
                all_valid = False
        else:
            print(f"  ERROR: Could not compute blur metrics")
            all_valid = False
        
        print()
    
    print("=" * 60)
    if all_valid:
        print("Status: All objects processed successfully")
    else:
        print("Status: ERRORS DETECTED - Blur artifacts present")
    
    return 0 if all_valid else 1

if __name__ == '__main__':
    sys.exit(main())