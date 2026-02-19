#!/usr/bin/env python3

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, classification_report
import json

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('/workspace/transactions.csv')

# Analyze class distribution
print("\n=== Class Distribution Analysis ===")
fraud_count = df['is_fraud'].sum()
total_count = len(df)
imbalance_ratio = fraud_count / total_count
print(f"Total transactions: {total_count}")
print(f"Fraudulent transactions: {fraud_count}")
print(f"Legitimate transactions: {total_count - fraud_count}")
print(f"Imbalance ratio: {imbalance_ratio:.4f} ({imbalance_ratio*100:.2f}%)")

# Prepare features and target
feature_columns = ['amount', 'merchant_category', 'time_of_day', 'day_of_week', 
                   'card_present', 'international']
X = df[feature_columns]
y = df['is_fraud']

# Train-test split
print("\n=== Splitting data ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Fraud cases in train: {y_train.sum()}")
print(f"Fraud cases in test: {y_test.sum()}")

# Calculate scale_pos_weight to handle class imbalance
# scale_pos_weight = (number of negative samples) / (number of positive samples)
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count
print(f"\n=== Handling Class Imbalance ===")
print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

# Train improved XGBoost model with class imbalance handling
print("\n=== Training improved XGBoost model ===")
model = xgb.XGBClassifier(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=200,
    scale_pos_weight=scale_pos_weight,  # Key parameter for handling imbalance
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1
)

model.fit(X_train, y_train)
print("Model training completed!")

# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate metrics
print("\n=== Model Performance on Test Set ===")
recall_fraud = recall_score(y_test, y_pred, pos_label=1)
precision_fraud = precision_score(y_test, y_pred, pos_label=1)

print(f"Recall (Fraud class): {recall_fraud:.4f} ({recall_fraud*100:.2f}%)")
print(f"Precision (Fraud class): {precision_fraud:.4f} ({precision_fraud*100:.2f}%)")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

# Check if metrics meet requirements
print("\n=== Requirements Check ===")
recall_met = recall_fraud >= 0.70
precision_met = precision_fraud >= 0.85
print(f"Recall >= 0.70: {'✓ PASS' if recall_met else '✗ FAIL'} ({recall_fraud:.4f})")
print(f"Precision >= 0.85: {'✓ PASS' if precision_met else '✗ FAIL'} ({precision_fraud:.4f})")

if recall_met and precision_met:
    print("\n✓ All requirements met!")
else:
    print("\n✗ Some requirements not met. Consider adjusting model parameters.")

# Save results to JSON
results = {
    "test_recall_fraud": round(recall_fraud, 2),
    "test_precision_fraud": round(precision_fraud, 2),
    "imbalance_ratio": round(imbalance_ratio, 2)
}

with open('/workspace/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n=== Results saved to /workspace/results.json ===")
print(json.dumps(results, indent=2))