#!/usr/bin/env python3

import numpy as np
import time
import math

def rastrigin(x):
    """Calculate the Rastrigin function value."""
    n = len(x)
    return 10 * n + sum(xi**2 - 10 * math.cos(2 * math.pi * xi) for xi in x)

def test_optimize_rastrigin():
    """Test the optimize_rastrigin function multiple times."""
    print("=" * 70)
    print("Testing optimize_rastrigin function")
    print("=" * 70)
    
    try:
        from optimizer import optimize_rastrigin
    except ImportError as e:
        print(f"ERROR: Failed to import optimize_rastrigin: {e}")
        return False
    
    num_runs = 10
    success_count = 0
    results = []
    times = []
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}:")
        print("-" * 50)
        
        start_time = time.time()
        try:
            result = optimize_rastrigin(dimensions=5, bounds=(-5.12, 5.12))
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            
            # Verify return type
            assert isinstance(result, tuple), f"Return value must be a tuple, got {type(result)}"
            assert len(result) == 2, f"Return tuple must have 2 elements, got {len(result)}"
            
            best_position, best_value = result
            
            # Verify best_position
            assert isinstance(best_position, (list, np.ndarray)), \
                f"best_position must be a list or array, got {type(best_position)}"
            best_position = list(best_position)
            assert len(best_position) == 5, \
                f"best_position must have 5 elements, got {len(best_position)}"
            
            # Verify all coordinates are floats and within bounds
            for i, coord in enumerate(best_position):
                assert isinstance(coord, (int, float, np.number)), \
                    f"Coordinate {i} must be numeric, got {type(coord)}"
                coord = float(coord)
                best_position[i] = coord
                assert -5.12 <= coord <= 5.12, \
                    f"Coordinate {i} = {coord} is out of bounds [-5.12, 5.12]"
            
            # Verify best_value
            assert isinstance(best_value, (int, float, np.number)), \
                f"best_value must be numeric, got {type(best_value)}"
            best_value = float(best_value)
            
            # Calculate actual function value
            actual_value = rastrigin(best_position)
            
            # Verify reported value matches actual value
            value_diff = abs(actual_value - best_value)
            assert value_diff < 1e-6, \
                f"Reported value {best_value} doesn't match actual {actual_value} (diff: {value_diff})"
            
            # Verify time constraint
            assert elapsed_time < 60, \
                f"Optimization took {elapsed_time:.2f}s, exceeds 60s limit"
            
            # Check if this run was successful (value <= 0.1)
            is_success = best_value <= 0.1
            if is_success:
                success_count += 1
            
            results.append({
                'position': best_position,
                'value': best_value,
                'time': elapsed_time,
                'success': is_success
            })
            
            print(f"  Position: {[f'{x:.6f}' for x in best_position]}")
            print(f"  Value: {best_value:.6f}")
            print(f"  Time: {elapsed_time:.2f}s")
            print(f"  Success: {'✓' if is_success else '✗'} (target: ≤ 0.1)")
            
        except AssertionError as e:
            print(f"  FAILED: {e}")
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            results.append({
                'position': None,
                'value': float('inf'),
                'time': elapsed_time,
                'success': False
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            results.append({
                'position': None,
                'value': float('inf'),
                'time': elapsed_time,
                'success': False
            })
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    success_rate = (success_count / num_runs) * 100
    valid_results = [r['value'] for r in results if r['value'] != float('inf')]
    
    if valid_results:
        best_value_found = min(valid_results)
        avg_value = sum(valid_results) / len(valid_results)
        print(f"Success rate: {success_count}/{num_runs} ({success_rate:.1f}%)")
        print(f"Best value found: {best_value_found:.6f}")
        print(f"Average value: {avg_value:.6f}")
        print(f"Average time: {sum(times)/len(times):.2f}s")
    else:
        print("No valid results obtained")
        return False
    
    # Check if success rate meets requirement
    required_success_rate = 80.0
    if success_rate >= required_success_rate:
        print(f"\n✓ SUCCESS: Success rate {success_rate:.1f}% meets requirement (≥ {required_success_rate}%)")
        return True
    else:
        print(f"\n✗ FAILURE: Success rate {success_rate:.1f}% below requirement (≥ {required_success_rate}%)")
        return False

if __name__ == "__main__":
    success = test_optimize_rastrigin()
    exit(0 if success else 1)
```

```python
#!/usr/bin/env python3

import numpy as np
import random

def optimize_rastrigin(dimensions=5, bounds=(-5.12, 5.12)):
    """
    Find the minimum of the Rastrigin function using Differential Evolution.
    
    Returns:
        tuple: (best_position, best_value)
        - best_position: list of floats representing the coordinates
        - best_value: float representing f(best_position)
    """
    
    def rastrigin(x):
        """Calculate the Rastrigin function value."""
        n = len(x)
        return 10 * n + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)
    
    # Differential Evolution parameters
    pop_size = 50
    max_iterations = 1000
    F = 0.8  # Differential weight
    CR = 0.9  # Crossover probability
    
    lower_bound, upper_bound = bounds
    
    # Initialize population randomly
    population = np.random.uniform(lower_bound, upper_bound, (pop_size, dimensions))
    fitness = np.array([rastrigin(ind) for ind in population])
    
    # Track best solution
    best_idx = np.argmin(fitness)
    best_position = population[best_idx].copy()
    best_value = fitness[best_idx]
    
    # Evolution loop
    for iteration in range(max_iterations):
        for i in range(pop_size):
            # Select three random individuals (different from current)
            indices = [idx for idx in range(pop_size) if idx != i]
            a, b, c = random.sample(indices, 3)
            
            # Mutation: V = Xa + F * (Xb - Xc)
            mutant = population[a] + F * (population[b] - population[c])
            
            # Clip to bounds
            mutant = np.clip(mutant, lower_bound, upper_bound)
            
            # Crossover
            trial = np.copy(population[i])
            cross_points = np.random.rand(dimensions) < CR
            
            # Ensure at least one dimension is from mutant
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            
            trial[cross_points] = mutant[cross_points]
            
            # Selection
            trial_fitness = rastrigin(trial)
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                
                # Update global best
                if trial_fitness < best_value:
                    best_value = trial_fitness
                    best_position = trial.copy()
        
        # Early stopping if we found a very good solution
        if best_value < 0.01:
            break
    
    return (best_position.tolist(), float(best_value))