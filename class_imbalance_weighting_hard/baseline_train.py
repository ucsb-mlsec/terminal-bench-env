#!/usr/bin/env python3
"""
Baseline fraud detection training script.
This is a NAIVE approach that does NOT handle class imbalance.
Expected to have poor fraud detection performance.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

# Set random seed for reproducibility
np.random.seed(42)

# Load training data
print("Loading training data...")
train_data = pd.read_csv('/data/transactions_train.csv')

# Separate features and target
X_train = train_data[[f'feature_{i}' for i in range(1, 21)]]
y_train = train_data['fraud']

print(f"Training set size: {len(y_train)}")
print(f"Fraud cases in training: {y_train.sum()} ({y_train.mean()*100:.2f}%)")

# Train baseline model WITHOUT any class imbalance handling
# No class_weight parameter, no resampling, no SMOTE
print("\nTraining baseline model (no imbalance handling)...")
baseline_model = LogisticRegression(random_state=42, max_iter=1000)
baseline_model.fit(X_train, y_train)

# Load test data
print("\nLoading test data...")
test_data = pd.read_csv('/data/transactions_test.csv')
X_test = test_data[[f'feature_{i}' for i in range(1, 21)]]
y_test = test_data['fraud']

print(f"Test set size: {len(y_test)}")
print(f"Fraud cases in test: {y_test.sum()} ({y_test.mean()*100:.2f}%)")

# Make predictions
print("\nMaking predictions...")
y_pred = baseline_model.predict(X_test)

# Calculate metrics for fraud class (positive class = 1)
baseline_recall = recall_score(y_test, y_pred, pos_label=1)
baseline_precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)

print("\n" + "="*50)
print("BASELINE MODEL PERFORMANCE (Fraud Detection)")
print("="*50)
print(f"Recall (Fraud):    {baseline_recall:.2f}")
print(f"Precision (Fraud): {baseline_precision:.2f}")
print("\nNote: This baseline approach ignores class imbalance,")
print("resulting in poor fraud detection performance.")