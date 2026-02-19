#!/usr/bin/env python3

import h2o
from h2o.automl import H2OAutoML
import os

def main():
    # Initialize H2O cluster
    h2o.init()
    
    # Load the iris dataset
    iris_path = "/workspace/ml_project/iris.csv"
    iris_data = h2o.import_file(iris_path)
    
    # Set the target column and features
    target = "species"
    features = [col for col in iris_data.columns if col != target]
    
    # Convert target to factor for classification
    iris_data[target] = iris_data[target].asfactor()
    
    # Split data into train and validation sets (optional but good practice)
    train, valid = iris_data.split_frame(ratios=[0.8], seed=42)
    
    # Set up AutoML
    aml = H2OAutoML(
        max_runtime_secs=30,
        seed=42,
        project_name="iris_classification"
    )
    
    # Train AutoML models
    aml.train(
        x=features,
        y=target,
        training_frame=train,
        validation_frame=valid
    )
    
    # Get the best model
    best_model = aml.leader
    
    # Create models directory if it doesn't exist
    models_dir = "/workspace/ml_project/models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the best model
    model_path = h2o.save_model(model=best_model, path=models_dir, force=True)
    
    # Rename to the expected filename
    expected_path = "/workspace/ml_project/models/iris_model.zip"
    if model_path != expected_path:
        import shutil
        shutil.move(model_path, expected_path)
    
    # Get leaderboard information
    leaderboard = aml.leaderboard
    models_trained = leaderboard.nrows
    
    # Get AUC score - for multiclass, use mean per class AUC
    best_auc = best_model.auc(valid=True)
    
    # Write result file
    result_path = "/workspace/ml_project/result.txt"
    with open(result_path, 'w') as f:
        f.write(f"STATUS: SUCCESS\n")
        f.write(f"MODELS_TRAINED: {models_trained}\n")
        f.write(f"BEST_AUC: {best_auc:.3f}\n")
    
    print(f"Training completed successfully!")
    print(f"Models trained: {models_trained}")
    print(f"Best model AUC: {best_auc:.3f}")
    print(f"Model saved to: {expected_path}")
    
    # Shutdown H2O cluster
    h2o.cluster().shutdown()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during training: {str(e)}")
        # Write failure status
        result_path = "/workspace/ml_project/result.txt"
        with open(result_path, 'w') as f:
            f.write(f"STATUS: FAILED\n")
            f.write(f"MODELS_TRAINED: 0\n")
            f.write(f"BEST_AUC: 0.000\n")
        raise