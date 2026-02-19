#!/usr/bin/env python3

import numpy as np
import os

# Create directories if they don't exist
os.makedirs('/workspace/data', exist_ok=True)

# Generate synthetic data if not exists
if not os.path.exists('/workspace/data/features.npy'):
    np.random.seed(42)
    features = np.random.randn(10000, 100).astype(np.float32)
    np.save('/workspace/data/features.npy', features)
    
if not os.path.exists('/workspace/data/labels.npy'):
    np.random.seed(42)
    labels = np.random.randint(0, 10, size=10000)
    labels_onehot = np.zeros((10000, 10), dtype=np.float32)
    labels_onehot[np.arange(10000), labels] = 1
    np.save('/workspace/data/labels.npy', labels_onehot)

# Load training data
print("Loading training data...")
X_train = np.load('/workspace/data/features.npy')
y_train = np.load('/workspace/data/labels.npy')

print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")

# Network architecture: 100 -> 64 -> 32 -> 10
input_size = 100
hidden1_size = 64
hidden2_size = 32
output_size = 10

# Initialize weights and biases
np.random.seed(123)
W1 = np.random.randn(input_size, hidden1_size).astype(np.float32) * 0.01
b1 = np.zeros((1, hidden1_size), dtype=np.float32)

W2 = np.random.randn(hidden1_size, hidden2_size).astype(np.float32) * 0.01
b2 = np.zeros((1, hidden2_size), dtype=np.float32)

W3 = np.random.randn(hidden2_size, output_size).astype(np.float32) * 0.01
b3 = np.zeros((1, output_size), dtype=np.float32)

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(np.float32)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), np.argmax(y_true, axis=1)] + 1e-7)
    loss = np.sum(log_likelihood) / m
    return loss

# Training parameters
batch_size = 1000
learning_rate = 0.01
epochs = 10
n_samples = X_train.shape[0]
n_batches = n_samples // batch_size

print(f"\nTraining configuration:")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {learning_rate}")
print(f"Epochs: {epochs}")
print(f"Number of batches per epoch: {n_batches}")

# Training loop
for epoch in range(epochs):
    epoch_loss = 0.0
    
    # Shuffle data at the beginning of each epoch
    indices = np.random.permutation(n_samples)
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    
    for batch_idx in range(n_batches):
        # Extract batch
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        
        X_batch = X_shuffled[start_idx:end_idx]
        y_batch = y_shuffled[start_idx:end_idx]
        
        # Forward propagation
        z1 = np.dot(X_batch, W1) + b1
        a1 = relu(z1)
        
        z2 = np.dot(a1, W2) + b2
        a2 = relu(z2)
        
        z3 = np.dot(a2, W3) + b3
        a3 = softmax(z3)
        
        # Compute loss
        batch_loss = cross_entropy_loss(a3, y_batch)
        epoch_loss += batch_loss
        
        # Backpropagation
        m = X_batch.shape[0]
        
        # Output layer gradients
        dz3 = a3 - y_batch
        dW3 = np.dot(a2.T, dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m
        
        # Hidden layer 2 gradients
        da2 = np.dot(dz3, W3.T)
        dz2 = da2 * relu_derivative(z2)
        dW2 = np.dot(a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer 1 gradients
        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * relu_derivative(z1)
        dW1 = np.dot(X_batch.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights and biases
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3
        
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
    
    avg_loss = epoch_loss / n_batches
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

print("\nTraining completed!")

# Save final model weights
model_weights = {
    'W1': W1,
    'b1': b1,
    'W2': W2,
    'b2': b2,
    'W3': W3,
    'b3': b3
}

np.save('/workspace/final_model.npy', model_weights)
print("Model weights saved to /workspace/final_model.npy")