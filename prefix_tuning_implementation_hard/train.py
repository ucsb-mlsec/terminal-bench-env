#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from model import PrefixTuning

# Set random seed for reproducibility
torch.manual_seed(42)

# Load and prepare data
def load_data(filepath):
    with open(filepath, 'r') as f:
        text = f.read()
    return text

def create_tokenizer(text):
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char, len(chars)

def encode(text, char_to_idx):
    return [char_to_idx[ch] for ch in text]

def decode(indices, idx_to_char):
    return ''.join([idx_to_char[i] for i in indices])

# Load data
text = load_data('data.txt')
char_to_idx, idx_to_char, vocab_size = create_tokenizer(text)
encoded_data = encode(text, char_to_idx)

print(f"Data loaded: {len(text)} characters, {vocab_size} unique characters")

# Model parameters
embedding_dim = 64
hidden_dim = 128
prefix_length = 10
seq_length = 20
batch_size = 8

# Initialize model
model = PrefixTuning(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    prefix_length=prefix_length
)

# THE BUG: Initialize optimizer with only the base model parameters, excluding prefix
# This means the prefix parameters will never be updated!
optimizer = optim.Adam(
    [p for name, p in model.named_parameters() if 'prefix' not in name],
    lr=0.01
)

criterion = nn.CrossEntropyLoss()

# Training function
def get_batch(data, seq_length, batch_size):
    """Sample random sequences from the data"""
    batch_x = []
    batch_y = []
    
    for _ in range(batch_size):
        start_idx = torch.randint(0, len(data) - seq_length - 1, (1,)).item()
        x = data[start_idx:start_idx + seq_length]
        y = data[start_idx + 1:start_idx + seq_length + 1]
        batch_x.append(x)
        batch_y.append(y)
    
    return torch.tensor(batch_x), torch.tensor(batch_y)

# Training loop
print("\nStarting training...")
print("=" * 50)

num_iterations = 50
initial_loss = None
final_loss = None

for iteration in range(1, num_iterations + 1):
    model.train()
    
    # Get batch
    x_batch, y_batch = get_batch(encoded_data, seq_length, batch_size)
    
    # Forward pass
    optimizer.zero_grad()
    output = model(x_batch)
    
    # Compute loss
    loss = criterion(output.view(-1, vocab_size), y_batch.view(-1))
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Store initial and final loss
    if iteration == 1:
        initial_loss = loss.item()
    if iteration == num_iterations:
        final_loss = loss.item()
    
    # Print progress
    print(f"Iteration {iteration}: Loss = {loss.item():.3f}")

print("=" * 50)
print("Training completed!")

# Write results to file
with open('/workspace/result.txt', 'w') as f:
    f.write(f"initial_loss={initial_loss:.3f}\n")
    f.write(f"final_loss={final_loss:.3f}\n")

print(f"\nResults written to result.txt")
print(f"Initial loss: {initial_loss:.3f}")
print(f"Final loss: {final_loss:.3f}")
print(f"Loss reduction: {initial_loss - final_loss:.3f}")