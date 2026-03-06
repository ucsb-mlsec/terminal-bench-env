#!/usr/bin/env python3

import random
import time
import json

def simulate_training(run_id, learning_rate, batch_size, epochs):
    """
    Simulates training of a machine learning model.
    Returns the final accuracy achieved.
    """
    print(f"\nStarting experiment {run_id} with lr={learning_rate}, batch_size={batch_size}, epochs={epochs}")
    
    # Initial values
    loss = 2.0
    accuracy = 0.5
    
    # Simulate training over epochs
    for epoch in range(1, epochs + 1):
        # Simulate loss decrease with some randomness
        loss_decrease = random.uniform(0.1, 0.2) * (learning_rate * 100)
        loss = max(0.3, loss - loss_decrease + random.uniform(-0.05, 0.05))
        
        # Simulate accuracy increase with some randomness
        accuracy_increase = random.uniform(0.02, 0.05)
        accuracy = min(0.95, accuracy + accuracy_increase + random.uniform(-0.01, 0.02))
        
        # Print progress
        print(f"Epoch {epoch}/{epochs} - Loss: {loss:.2f}, Accuracy: {accuracy:.2f}")
        
        # Simulate computation time
        time.sleep(0.1)
    
    # Final accuracy with some variation based on hyperparameters
    final_accuracy = accuracy + random.uniform(-0.03, 0.03)
    final_accuracy = max(0.75, min(0.95, final_accuracy))
    
    print(f"Experiment {run_id} completed. Final accuracy: {final_accuracy:.2f}")
    
    return final_accuracy


def main():
    """
    Main execution: runs multiple experiments with different configurations.
    """
    # Define experiment configurations
    configurations = [
        {"learning_rate": 0.001, "batch_size": 32, "epochs": 10},
        {"learning_rate": 0.01, "batch_size": 64, "epochs": 5},
        {"learning_rate": 0.005, "batch_size": 16, "epochs": 15},
        {"learning_rate": 0.1, "batch_size": 128, "epochs": 8},
        {"learning_rate": 0.001, "batch_size": 64, "epochs": 20},
    ]
    
    experiment_results = []
    
    # Run experiments
    for idx, config in enumerate(configurations, start=1):
        run_id = f"run_{idx:03d}"
        
        # Run the training simulation
        final_accuracy = simulate_training(
            run_id=run_id,
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            epochs=config["epochs"]
        )
        
        # Store experiment data
        experiment_record = {
            "run_id": run_id,
            "config": {
                "learning_rate": config["learning_rate"],
                "batch_size": config["batch_size"],
                "epochs": config["epochs"]
            },
            "final_accuracy": round(final_accuracy, 4)
        }
        
        experiment_results.append(experiment_record)
        
        # Small delay between experiments
        time.sleep(0.2)
    
    # Save all results to JSON file
    output_path = "/home/experiments.json"
    with open(output_path, 'w') as f:
        json.dump(experiment_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"All experiments completed!")
    print(f"Results saved to {output_path}")
    print(f"Total experiments: {len(experiment_results)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()