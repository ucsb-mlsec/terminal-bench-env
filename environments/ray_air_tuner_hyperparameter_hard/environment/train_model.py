#!/usr/bin/env python3

import ray
from ray import tune
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Initial poorly configured training script that needs optimization
# Issues:
# 1. Search space is too narrow
# 2. No early stopping configured
# 3. Using basic sampling (no HyperOpt/Optuna)
# 4. Too few trials (num_samples=5)
# 5. Suboptimal resource allocation
# 6. No proper result saving

def train_model(config):
    """
    Train a Ridge regression model with given hyperparameters.
    Reports validation MSE back to Ray Tune.
    """
    # Load housing data
    df = pd.read_csv('/workspace/housing_data.csv')
    
    # Split features and target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Train-test split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Ridge regression model with alpha from config
    model = Ridge(alpha=config['alpha'], random_state=42)
    model.fit(X_train, y_train)
    
    # Predict and calculate MSE on validation set
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    
    # Report MSE to Ray Tune
    tune.report(mse=mse)


if __name__ == '__main__':
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # POORLY CONFIGURED: Search space is too narrow (0.00001 to 0.0001)
    # This range is inadequate for finding optimal regularization
    config = {
        'alpha': tune.uniform(0.00001, 0.0001)  # Too narrow range!
    }
    
    # Run hyperparameter tuning with poor configuration:
    # - No scheduler (no early stopping)
    # - Only 5 samples (too few trials)
    # - Basic sampling (no advanced search algorithms)
    # - Suboptimal resource allocation
    analysis = tune.run(
        train_model,
        config=config,
        num_samples=5,  # Too few trials!
        resources_per_trial={'cpu': 4, 'gpu': 0},  # Suboptimal allocation
        verbose=1
    )
    
    # Just print results, no proper saving
    print("Best config:", analysis.best_config)
    print("Best MSE:", analysis.best_result['mse'])