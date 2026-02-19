#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

class MemoryBuffer:
    """Memory buffer to store representative samples from previous batches."""
    
    def __init__(self, max_size=300):
        self.max_size = max_size
        self.samples = []
        self.labels = []
        
    def add_batch(self, X, y):
        """Add samples from a batch to memory using reservoir sampling."""
        n_current = len(self.samples)
        n_new = len(X)
        
        # Calculate how many samples to keep from this batch
        if n_current == 0:
            # First batch - take up to max_size samples
            n_to_keep = min(n_new, self.max_size)
            indices = np.random.choice(n_new, n_to_keep, replace=False)
            self.samples = X.iloc[indices].values.tolist()
            self.labels = y.iloc[indices].values.tolist()
        else:
            # Subsequent batches - maintain proportional representation
            # Add new samples and then downsample if needed
            new_samples = X.values.tolist()
            new_labels = y.values.tolist()
            
            self.samples.extend(new_samples)
            self.labels.extend(new_labels)
            
            # If we exceed max_size, randomly sample to reduce
            if len(self.samples) > self.max_size:
                indices = np.random.choice(len(self.samples), self.max_size, replace=False)
                self.samples = [self.samples[i] for i in indices]
                self.labels = [self.labels[i] for i in indices]
    
    def get_memory(self):
        """Return memory samples as arrays."""
        if len(self.samples) == 0:
            return None, None
        return np.array(self.samples), np.array(self.labels)
    
    def size(self):
        """Return current memory size."""
        return len(self.samples)


def load_batch(filepath):
    """Load a batch from CSV file."""
    df = pd.read_csv(filepath)
    feature_cols = ['age', 'income', 'purchase_history', 'engagement_score']
    X = df[feature_cols]
    y = df['will_purchase']
    return X, y


def train_with_memory(batch_files, max_memory_size=300):
    """
    Train model sequentially on batches using memory mechanism.
    
    Args:
        batch_files: List of file paths to batch CSV files
        max_memory_size: Maximum number of samples to store in memory
    
    Returns:
        Trained model and list of (X, y) tuples for each batch
    """
    memory = MemoryBuffer(max_size=max_memory_size)
    model = LogisticRegression(max_iter=1000, random_state=42)
    
    # Store all batches for later evaluation
    all_batches = []
    
    for i, batch_file in enumerate(batch_files):
        print(f"\nTraining on batch {i+1}: {batch_file}")
        
        # Load current batch
        X_batch, y_batch = load_batch(batch_file)
        all_batches.append((X_batch, y_batch))
        
        # Combine current batch with memory
        if memory.size() > 0:
            X_memory, y_memory = memory.get_memory()
            X_train = pd.DataFrame(
                np.vstack([X_batch.values, X_memory]),
                columns=X_batch.columns
            )
            y_train = pd.Series(np.hstack([y_batch.values, y_memory]))
            print(f"Training with {len(X_batch)} new samples + {memory.size()} memory samples")
        else:
            X_train = X_batch
            y_train = y_batch
            print(f"Training with {len(X_batch)} new samples (no memory yet)")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Add current batch to memory (before moving to next batch)
        memory.add_batch(X_batch, y_batch)
        print(f"Memory size after update: {memory.size()}")
    
    return model, all_batches


def evaluate_model(model, batches):
    """
    Evaluate model on all batches.
    
    Args:
        model: Trained model
        batches: List of (X, y) tuples
    
    Returns:
        List of accuracy scores
    """
    accuracies = []
    for i, (X, y) in enumerate(batches):
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        accuracies.append(accuracy)
        print(f"Batch {i+1} accuracy: {accuracy:.4f}")
    return accuracies


def main():
    print("=" * 60)
    print("Continual Learning with Memory Mechanism")
    print("=" * 60)
    
    # Define batch files
    batch_files = [
        '/data/batch_1.csv',
        '/data/batch_2.csv',
        '/data/batch_3.csv'
    ]
    
    # Train model with memory mechanism
    model, all_batches = train_with_memory(batch_files, max_memory_size=300)
    
    print("\n" + "=" * 60)
    print("Final Model Evaluation on All Batches")
    print("=" * 60)
    
    # Evaluate on all batches
    accuracies = evaluate_model(model, all_batches)
    
    # Create solution directory if it doesn't exist
    os.makedirs('/solution', exist_ok=True)
    
    # Save results
    output_path = '/solution/performance.txt'
    with open(output_path, 'w') as f:
        f.write(f"batch1_accuracy={accuracies[0]:.2f}\n")
        f.write(f"batch2_accuracy={accuracies[1]:.2f}\n")
        f.write(f"batch3_accuracy={accuracies[2]:.2f}\n")
    
    print(f"\nResults saved to {output_path}")
    print("\nSummary:")
    print(f"  Batch 1 Accuracy: {accuracies[0]:.2f}")
    print(f"  Batch 2 Accuracy: {accuracies[1]:.2f}")
    print(f"  Batch 3 Accuracy: {accuracies[2]:.2f}")
    
    # Check success criteria
    all_above_threshold = all(acc > 0.65 for acc in accuracies)
    if all_above_threshold:
        print("\n✓ SUCCESS: All accuracies above 0.65 threshold!")
    else:
        print("\n✗ WARNING: Some accuracies below 0.65 threshold")


if __name__ == "__main__":
    main()