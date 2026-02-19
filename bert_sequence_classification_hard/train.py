#!/usr/bin/env python3

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import json
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load data
df = pd.read_csv('/workspace/data/feedback.csv')

# BUG 1: Reversed label encoding - this will cause predictions to be inverted
label_map = {
    'positive': 2,  # Should be 0
    'neutral': 1,   # Correct
    'negative': 0   # Should be 2
}

# BUG 2: Reversed decode map - compounds the label problem
label_decode = {
    2: 'positive',  # Should be 0: 'positive'
    1: 'neutral',   # Correct
    0: 'negative'   # Should be 2: 'negative'
}

# Encode labels
df['label_encoded'] = df['label'].map(label_map)

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Custom Dataset
class FeedbackDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create datasets
train_dataset = FeedbackDataset(
    train_df['text'].reset_index(drop=True),
    train_df['label_encoded'].reset_index(drop=True),
    tokenizer
)

test_dataset = FeedbackDataset(
    test_df['text'].reset_index(drop=True),
    test_df['label_encoded'].reset_index(drop=True),
    tokenizer
)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Setup training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

# Evaluation
model.eval()
correct = 0
total = 0
all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # BUG 3: Using wrong index for predictions - using argmin instead of argmax
        predictions = torch.argmin(outputs.logits, dim=1)
        
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = correct / total
print(f'Test Accuracy: {test_accuracy:.4f}')

# Save results
results = {
    'test_accuracy': round(test_accuracy, 4),
    'bugs_fixed': 0  # Will be updated after bugs are fixed
}

os.makedirs('/workspace', exist_ok=True)
with open('/workspace/solution.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'Results saved to /workspace/solution.json')
print(f'Test accuracy: {test_accuracy:.4f}')