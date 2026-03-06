#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import yaml
import random
from torch.utils.data import Dataset, DataLoader
from model import LSTMClassifier

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert text to indices
        tokens = text.lower().split()
        indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Pad or truncate
        if len(indices) < self.max_len:
            indices += [self.vocab['<PAD>']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
            
        return torch.LongTensor(indices), torch.FloatTensor([label])

def build_vocab(texts, min_freq=2):
    word_freq = {}
    for text in texts:
        for word in text.lower().split():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    
    return vocab

def main():
    # Load configuration
    with open('/workspace/text_classifier/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    max_iterations = config.get('max_iterations', 150)
    
    print(f"Loading data...")
    # Load data
    df = pd.read_csv('/workspace/text_classifier/data/reviews.csv')
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    # Build vocabulary
    print(f"Building vocabulary...")
    vocab = build_vocab(texts)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create dataset and dataloader
    dataset = ReviewDataset(texts, labels, vocab)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    embed_dim = 128
    hidden_dim = 256
    model = LSTMClassifier(vocab_size, embed_dim, hidden_dim)
    
    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"Starting training for {max_iterations} iterations...")
    
    # Training loop
    iteration = 0
    model.train()
    
    while iteration < max_iterations:
        for batch_texts, batch_labels in dataloader:
            if iteration >= max_iterations:
                break
            
            # Forward pass
            outputs = model(batch_texts)
            loss = criterion(outputs, batch_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # BUG: No gradient clipping here - this causes NaN loss due to exploding gradients
            
            optimizer.step()
            
            iteration += 1
            
            # Print progress
            if iteration % 10 == 0:
                print(f"Iteration {iteration}/{max_iterations}, Loss: {loss.item():.4f}")
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"ERROR: Loss became NaN at iteration {iteration}!")
                return
    
    print(f"\nTraining completed successfully!")
    print(f"Final loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()