#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import json
import numpy as np
import os

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
DATA_PATH = 'data/samples.jsonl'
MODEL_NAME = 'prajjwal1/bert-tiny'
SOFT_PROMPT_LENGTH = 8
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 0.01

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# Load data
print("Loading data...")
texts = []
labels = []
with open(DATA_PATH, 'r') as f:
    for line in f:
        data = json.loads(line)
        texts.append(data['text'])
        labels.append(data['label'])

print(f"Loaded {len(texts)} samples")

# Load tokenizer and model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModel.from_pretrained(MODEL_NAME)

# BUG 1: Forgot to freeze base model parameters!
# All base model params will be trained, defeating the purpose

# Get embedding dimension
embedding_dim = base_model.config.hidden_size

# BUG 2: Initialize soft prompts but with wrong shape
# Should be (SOFT_PROMPT_LENGTH, embedding_dim) but initialized as (embedding_dim, SOFT_PROMPT_LENGTH)
soft_prompts = nn.Parameter(torch.randn(embedding_dim, SOFT_PROMPT_LENGTH))

# Classification head
classifier = nn.Linear(embedding_dim, 2)

# BUG 3: Wrong optimizer - only optimizing classifier, forgetting soft_prompts!
optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)

# BUG 4: Using wrong loss function for classification
criterion = nn.MSELoss()  # Should be CrossEntropyLoss!

# Split data (no validation split, training on everything)
dataset = SentimentDataset(texts, labels)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Starting training...")
step_count = 0

for epoch in range(EPOCHS):
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_texts, batch_labels in dataloader:
        # Tokenize
        encoded = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
        
        # Get input embeddings
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        # Get embeddings from base model
        input_embeds = base_model.embeddings.word_embeddings(input_ids)
        
        # BUG 5: Wrong concatenation - shape mismatch!
        # soft_prompts has wrong shape, and not expanding for batch
        # This will likely crash or produce wrong results
        try:
            combined_embeds = torch.cat([soft_prompts, input_embeds], dim=1)
        except:
            # Fallback that will give wrong results
            combined_embeds = input_embeds
        
        # Update attention mask - but incorrectly!
        # BUG 6: Not actually extending attention mask for soft prompts
        
        # Forward pass through base model
        outputs = base_model(inputs_embeds=combined_embeds, attention_mask=attention_mask)
        
        # BUG 7: Using wrong pooling strategy
        # Just taking first token instead of proper pooling
        pooled = outputs.last_hidden_state[:, 0, :]
        
        # Classification
        logits = classifier(pooled)
        
        # BUG 8: Wrong label format for MSELoss
        loss = criterion(logits, batch_labels.float().unsqueeze(1))
        
        # BUG 9: Missing zero_grad!
        # Gradients will accumulate incorrectly
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy (incorrectly)
        preds = (logits[:, 0] > 0.5).long()
        correct += (preds == batch_labels).sum().item()
        total += len(batch_labels)
        step_count += 1
    
    accuracy = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

print("Training complete!")
print(f"Final accuracy: {accuracy:.4f}")
print(f"Total steps: {step_count}")

# Save results (but wrong format)
with open('results.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Steps: {step_count}\n")

# Try to save soft prompts
torch.save(soft_prompts, 'soft_prompts.pt')
print("Saved soft prompts")