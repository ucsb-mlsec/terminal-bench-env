#!/usr/bin/env python3

import json
import os
import glob
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_config():
    """Load joblib configuration from config file."""
    config_path = '/opt/ml_pipeline/joblib_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def train_single_model(dataset_path, model_id):
    """
    Train a single model on a dataset.
    
    Args:
        dataset_path: Path to the CSV dataset
        model_id: Identifier for the model
    
    Returns:
        Tuple of (model_id, trained_model)
    """
    try:
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Assume last column is target, rest are features
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model {model_id} trained with accuracy: {accuracy:.4f}")
        
        return (model_id, model)
    
    except Exception as e:
        print(f"Error training model {model_id}: {str(e)}")
        return (model_id, None)

def main():
    """Main training pipeline."""
    print("Starting training pipeline...")
    
    # Load configuration
    config = load_config()
    print(f"Configuration loaded: {config}")
    
    # Get dataset files
    dataset_dir = '/data/datasets/'
    dataset_files = sorted(glob.glob(os.path.join(dataset_dir, '*.csv')))
    
    if not dataset_files:
        print("No dataset files found!")
        return
    
    print(f"Found {len(dataset_files)} datasets to process")
    
    # Create output directory if it doesn't exist
    output_dir = '/models/output/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract joblib parameters from config
    n_jobs = config.get('n_jobs', -1)
    backend = config.get('backend', 'loky')
    verbose = config.get('verbose', 10)
    
    print(f"Starting parallel training with backend='{backend}', n_jobs={n_jobs}")
    
    # Train models in parallel
    results = joblib.Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        joblib.delayed(train_single_model)(dataset_path, i) 
        for i, dataset_path in enumerate(dataset_files, 1)
    )
    
    print("Training complete. Saving models...")
    
    # Save trained models
    saved_count = 0
    for model_id, model in results:
        if model is not None:
            model_path = os.path.join(output_dir, f'model_{model_id}.pkl')
            joblib.dump(model, model_path)
            saved_count += 1
    
    print(f"Models saved: {saved_count}/{len(results)} models successfully saved to {output_dir}")
    print("Pipeline execution complete!")

if __name__ == '__main__':
    main()