#!/usr/bin/env python3

import json
import numpy as np
from typing import Dict, Optional, List
import ray
from ray import tune
from ray.tune.search import Searcher
from ray.tune.search.sample import Domain

# Import the training function
import sys
sys.path.insert(0, '/workspace/ray_custom')
from training_function import train_model


class ExplorationSearcher(Searcher):
    """Custom searcher that explores underperforming regions of hyperparameter space."""
    
    def __init__(self, search_space: Dict, historical_results: List[Dict], max_trials: int = 30):
        super().__init__(metric="validation_accuracy", mode="max")
        self.search_space = search_space
        self.historical_results = historical_results
        self.max_trials = max_trials
        self.trial_count = 0
        self.suggested_configs = []
        
        # Identify underperforming regions (accuracy < 0.7)
        self.poor_regions = []
        for result in historical_results:
            if result.get('validation_accuracy', 1.0) < 0.7:
                self.poor_regions.append({
                    'learning_rate': result['config']['learning_rate'],
                    'batch_size': result['config']['batch_size'],
                    'dropout_rate': result['config']['dropout_rate']
                })
        
        # Track which regions we've explored
        self.explored_regions = set()
        
    def suggest(self, trial_id: str) -> Optional[Dict]:
        """Suggest a new trial configuration."""
        if self.trial_count >= self.max_trials:
            return None
        
        self.trial_count += 1
        
        # 80% of trials focus on poor regions, 20% random exploration
        if self.poor_regions and np.random.random() < 0.8:
            # Sample from a poor-performing region with some perturbation
            base_config = self.poor_regions[np.random.randint(len(self.poor_regions))]
            
            # Add region to explored set (discretized for counting)
            region_key = (
                round(base_config['learning_rate'], 3),
                base_config['batch_size'],
                round(base_config['dropout_rate'], 2)
            )
            self.explored_regions.add(region_key)
            
            # Perturb around this configuration
            config = {
                'learning_rate': np.clip(
                    base_config['learning_rate'] * np.random.uniform(0.7, 1.3),
                    0.0001, 0.1
                ),
                'batch_size': int(np.random.choice([16, 32, 64, 128])),
                'dropout_rate': np.clip(
                    base_config['dropout_rate'] + np.random.uniform(-0.1, 0.1),
                    0.1, 0.5
                )
            }
        else:
            # Random exploration
            config = {
                'learning_rate': np.random.uniform(0.0001, 0.1),
                'batch_size': int(np.random.choice([16, 32, 64, 128])),
                'dropout_rate': np.random.uniform(0.1, 0.5)
            }
        
        self.suggested_configs.append(config)
        return config
    
    def on_trial_complete(self, trial_id: str, result: Optional[Dict] = None, error: bool = False):
        """Called when a trial completes."""
        pass
    
    def save(self, checkpoint_path: str):
        """Save searcher state."""
        pass
    
    def restore(self, checkpoint_path: str):
        """Restore searcher state."""
        pass


def load_data():
    """Load search configuration and historical results."""
    with open('/workspace/ray_custom/search_config.json', 'r') as f:
        search_config = json.load(f)
    
    with open('/workspace/ray_custom/historical_results.json', 'r') as f:
        historical_results = json.load(f)
    
    return search_config, historical_results


def run_custom_search():
    """Execute the custom search with Ray Tune."""
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Load data
    search_config, historical_results = load_data()
    
    # Create search space for Ray Tune
    tune_search_space = {
        'learning_rate': tune.uniform(0.0001, 0.1),
        'batch_size': tune.choice([16, 32, 64, 128]),
        'dropout_rate': tune.uniform(0.1, 0.5)
    }
    
    # Initialize custom searcher
    custom_searcher = ExplorationSearcher(
        search_space=search_config,
        historical_results=historical_results,
        max_trials=30
    )
    
    # Configure Ray Tune
    tuner = tune.Tuner(
        train_model,
        tune_config=tune.TuneConfig(
            search_alg=custom_searcher,
            num_samples=30,
            max_concurrent_trials=4,
        ),
        param_space=tune_search_space,
    )
    
    # Run the search
    results = tuner.fit()
    
    # Collect results
    all_results = []
    for result in results:
        try:
            metrics = result.metrics
            config = result.config
            all_results.append({
                'validation_accuracy': metrics.get('validation_accuracy', 0.0),
                'config': config
            })
        except:
            pass
    
    # Calculate statistics
    total_trials = len(all_results)
    avg_accuracy = np.mean([r['validation_accuracy'] for r in all_results])
    exploration_regions = len(custom_searcher.explored_regions)
    
    # Count trials in poor-performing regions
    trials_in_poor_regions = 0
    for result in all_results:
        config = result['config']
        # Check if this trial is near any historical poor region
        for poor_region in custom_searcher.poor_regions:
            lr_close = abs(config['learning_rate'] - poor_region['learning_rate']) < 0.02
            bs_match = config['batch_size'] == poor_region['batch_size']
            dr_close = abs(config['dropout_rate'] - poor_region['dropout_rate']) < 0.15
            
            if (lr_close and bs_match) or (lr_close and dr_close) or (bs_match and dr_close):
                trials_in_poor_regions += 1
                break
    
    # Write summary
    with open('/workspace/ray_custom/search_summary.txt', 'w') as f:
        f.write(f"total_trials={total_trials}\n")
        f.write(f"avg_accuracy={avg_accuracy:.3f}\n")
        f.write(f"exploration_regions={exploration_regions}\n")
    
    print(f"\nSearch completed successfully!")
    print(f"Total trials: {total_trials}")
    print(f"Average accuracy: {avg_accuracy:.3f}")
    print(f"Exploration regions: {exploration_regions}")
    print(f"Trials exploring poor regions: {trials_in_poor_regions}/30")
    
    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    run_custom_search()