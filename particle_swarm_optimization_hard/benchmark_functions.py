#!/usr/bin/env python3
"""
Benchmark Functions for Optimization Testing

This module provides four well-known optimization benchmark functions commonly used
to test optimization algorithms. Each function is highly multi-modal with many
local minima, making them challenging test cases.
"""

import numpy as np
import math


def rastrigin(x):
    """
    Rastrigin function - highly multi-modal with many regularly distributed local minima.
    
    Formula: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
    
    Properties:
    - Global minimum: f(0, 0, ..., 0) = 0
    - Search domain: [-5.12, 5.12] for each dimension
    - Highly multi-modal with many local minima
    
    Args:
        x: numpy array or list representing an n-dimensional point
    
    Returns:
        float: function value at point x
    """
    x = np.asarray(x)
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def ackley(x):
    """
    Ackley function - nearly flat outer region with a large hole at the center.
    
    Formula: f(x) = -20*exp(-0.2*sqrt(sum(x_i^2)/n)) - exp(sum(cos(2*pi*x_i))/n) + 20 + e
    
    Properties:
    - Global minimum: f(0, 0, ..., 0) = 0
    - Search domain: [-32.768, 32.768] for each dimension
    - Characterized by a nearly flat outer region and a large hole at the center
    
    Args:
        x: numpy array or list representing an n-dimensional point
    
    Returns:
        float: function value at point x
    """
    x = np.asarray(x)
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    
    return term1 + term2 + 20 + np.e


def griewank(x):
    """
    Griewank function - product term introduces interdependence between variables.
    
    Formula: f(x) = sum(x_i^2/4000) - product(cos(x_i/sqrt(i+1))) + 1
    
    Properties:
    - Global minimum: f(0, 0, ..., 0) = 0
    - Search domain: [-600, 600] for each dimension
    - Product term introduces interdependence between variables
    
    Args:
        x: numpy array or list representing an n-dimensional point
    
    Returns:
        float: function value at point x
    """
    x = np.asarray(x)
    n = len(x)
    
    sum_term = np.sum(x**2) / 4000
    
    # Product term with i being 0-indexed
    prod_term = 1.0
    for i in range(n):
        prod_term *= np.cos(x[i] / np.sqrt(i + 1))
    
    return sum_term - prod_term + 1


def schwefel(x):
    """
    Schwefel function - deceptive with optimum far from second-best local minima.
    
    Formula: f(x) = 418.9829*n - sum(x_i*sin(sqrt(abs(x_i))))
    
    Properties:
    - Global minimum: f(420.9687, 420.9687, ..., 420.9687) = 0
    - Search domain: [-500, 500] for each dimension
    - Deceptive function where optimum is far from second-best local minima
    
    Args:
        x: numpy array or list representing an n-dimensional point
    
    Returns:
        float: function value at point x
    """
    x = np.asarray(x)
    n = len(x)
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))


# Dictionary mapping function names to their search bounds
BOUNDS = {
    'rastrigin': (-5.12, 5.12),
    'ackley': (-32.768, 32.768),
    'griewank': (-600, 600),
    'schwefel': (-500, 500)
}


# Dictionary mapping function names to their functions for easy access
FUNCTIONS = {
    'rastrigin': rastrigin,
    'ackley': ackley,
    'griewank': griewank,
    'schwefel': schwefel
}


# Known global minima positions
GLOBAL_MINIMA = {
    'rastrigin': 0.0,  # at [0, 0, ..., 0]
    'ackley': 0.0,     # at [0, 0, ..., 0]
    'griewank': 0.0,   # at [0, 0, ..., 0]
    'schwefel': 0.0    # at [420.9687, 420.9687, ..., 420.9687]
}


def get_optimum_position(function_name, dimension):
    """
    Get the known global optimum position for a given function and dimension.
    
    Args:
        function_name: name of the function ('rastrigin', 'ackley', 'griewank', 'schwefel')
        dimension: dimensionality of the problem
    
    Returns:
        numpy array: position of the global optimum
    """
    if function_name == 'schwefel':
        return np.full(dimension, 420.9687)
    else:
        return np.zeros(dimension)
```

Now I'll create the optimization solver:

```python
#!/usr/bin/env python3
"""
Optimization Engine for Multi-Modal Benchmark Functions

This module implements a hybrid optimization approach combining:
1. Differential Evolution for global exploration
2. Local refinement for exploitation
3. Adaptive parameter control
"""

import numpy as np
import json
from benchmark_functions import FUNCTIONS, BOUNDS, get_optimum_position


class DifferentialEvolution:
    """
    Differential Evolution optimizer with adaptive parameters.
    """
    
    def __init__(self, func, bounds, dim, pop_size=None, max_iter=3000, seed=42):
        """
        Initialize the DE optimizer.
        
        Args:
            func: objective function to minimize
            bounds: tuple of (lower, upper) bounds
            dim: dimensionality of the problem
            pop_size: population size (default: 10 * dim)
            max_iter: maximum iterations
            seed: random seed for reproducibility
        """
        self.func = func
        self.bounds = bounds
        self.dim = dim
        self.pop_size = pop_size if pop_size else max(50, 10 * dim)
        self.max_iter = max_iter
        self.seed = seed
        
        np.random.seed(seed)
        
        # Adaptive parameters
        self.F = 0.8  # mutation factor
        self.CR = 0.9  # crossover probability
        
    def initialize_population(self):
        """Initialize population uniformly within bounds."""
        lower, upper = self.bounds
        pop = np.random.uniform(lower, upper, (self.pop_size, self.dim))
        fitness = np.array([self.func(ind) for ind in pop])
        return pop, fitness
    
    def mutate(self, population, best_idx, iteration):
        """
        Create mutant vectors using DE/best/1/bin strategy with adaptive parameters.
        """
        mutants = []
        
        # Adaptive F: decreases as we converge
        F = self.F * (1 - 0.5 * iteration / self.max_iter)
        
        for i in range(self.pop_size):
            # Select three random distinct individuals
            indices = list(range(self.pop_size))
            indices.remove(i)
            
            # Use best individual for guidance
            a, b = np.random.choice(indices, 2, replace=False)
            
            # Mutation: v = best + F * (a - b)
            mutant = population[best_idx] + F * (population[a] - population[b])
            
            # Ensure bounds
            mutant = np.clip(mutant, self.bounds[0], self.bounds[1])
            mutants.append(mutant)
        
        return np.array(mutants)
    
    def crossover(self, population, mutants, iteration):
        """
        Perform binomial crossover with adaptive CR.
        """
        # Adaptive CR: increases as we converge for more exploitation
        CR = self.CR * (0.5 + 0.5 * iteration / self.max_iter)
        
        trials = []
        for i in range(self.pop_size):
            trial = np.copy(population[i])
            
            # Ensure at least one parameter from mutant
            j_rand = np.random.randint(self.dim)
            
            for j in range(self.dim):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = mutants[i][j]
            
            trials.append(trial)
        
        return np.array(trials)
    
    def optimize(self):
        """
        Run the optimization process.
        
        Returns:
            best_position, best_value
        """
        # Initialize
        population, fitness = self.initialize_population()
        best_idx = np.argmin(fitness)
        best_position = population[best_idx].copy()
        best_value = fitness[best_idx]
        
        stagnation_counter = 0
        prev_best = best_value
        
        for iteration in range(self.max_iter):
            # Mutation
            mutants = self.mutate(population, best_idx, iteration)
            
            # Crossover
            trials = self.crossover(population, mutants, iteration)
            
            # Selection
            for i in range(self.pop_size):
                trial_fitness = self.func(trials[i])
                
                if trial_fitness < fitness[i]:
                    population[i] = trials[i]
                    fitness[i] = trial_fitness
                    
                    if trial_fitness < best_value:
                        best_value = trial_fitness
                        best_position = trials[i].copy()
                        best_idx = i
            
            # Check for stagnation
            if abs(best_value - prev_best) < 1e-10:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            prev_best = best_value
            
            # Early stopping if converged
            if best_value < 1e-6:
                break
            
            # Restart if stagnated
            if stagnation_counter > 100:
                # Reinitialize half the population
                n_reinit = self.pop_size // 2
                lower, upper = self.bounds
                new_pop = np.random.uniform(lower, upper, (n_reinit, self.dim))
                population[-n_reinit:] = new_pop
                fitness[-n_reinit:] = np.array([self.func(ind) for ind in new_pop])
                stagnation_counter = 0
        
        return best_position, best_value
    
    def local_refinement(self, position):
        """
        Apply local refinement using Nelder-Mead-like approach.
        """
        from scipy.optimize import minimize
        
        result = minimize(
            self.func,
            position,
            method='Powell',
            bounds=[self.bounds] * self.dim,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        
        return result.x, result.fun


def optimize_function(func_name, dimension, max_attempts=3):
    """
    Optimize a specific function for given dimension with multiple attempts.
    
    Args:
        func_name: name of the benchmark function
        dimension: problem dimensionality
        max_attempts: maximum number of optimization attempts
    
    Returns:
        dict with 'position' and 'value' keys
    """
    func = FUNCTIONS[func_name]
    bounds = BOUNDS[func_name]
    
    best_position = None
    best_value = float('inf')
    
    for attempt in range(max_attempts):
        # Use different seed for each attempt
        seed = 42 + attempt * 1000
        
        # Create optimizer
        optimizer = DifferentialEvolution(
            func=func,
            bounds=bounds,
            dim=dimension,
            pop_size=max(50, 15 * dimension),
            max_iter=5000,
            seed=seed
        )
        
        # Run optimization
        position, value = optimizer.optimize()
        
        # Try local refinement
        try:
            position, value = optimizer.local_refinement(position)
        except:
            pass
        
        if value < best_value:
            best_value = value
            best_position = position.copy()
        
        # If we found a good enough solution, stop
        if best_value < 0.001:
            break
    
    return {
        'position': best_position.tolist(),
        'value': float(best_value)
    }


def main():
    """
    Main optimization routine for all benchmark functions and dimensions.
    """
    results = {}
    
    function_names = ['rastrigin', 'ackley', 'griewank', 'schwefel']
    dimensions = [10, 20, 30]
    
    for func_name in function_names:
        print(f"\nOptimizing {func_name}...")
        
        for dim in dimensions:
            print(f"  Dimension {dim}...", end=' ', flush=True)
            
            key = f"{func_name}_{dim}d"
            result = optimize_function(func_name, dim, max_attempts=3)
            results[key] = result
            
            print(f"Value: {result['value']:.6f}")
    
    # Save results
    with open('/workspace/optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*50)
    print("Optimization Complete!")
    print("="*50)
    
    # Verify all results meet criteria
    all_pass = True
    for key, result in results.items():
        if result['value'] > 0.01:
            print(f"WARNING: {key} did not meet criteria (value={result['value']:.6f})")
            all_pass = False
    
    if all_pass:
        print("✓ All optimization targets met (all values ≤ 0.01)")
    else:
        print("✗ Some targets not met")
    
    return results


if __name__ == '__main__':
    main()