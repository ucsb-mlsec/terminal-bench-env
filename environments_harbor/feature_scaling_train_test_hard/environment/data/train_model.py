"""
Training script for binary classification model.
Loads data, normalizes features, splits into train/test, and trains a logistic regression model.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("/home/user/ml_project/data.csv")

# Separate features and target
X = data.drop('target', axis=1).values
y = data['target'].values

# Normalize features using mean and standard deviation
# BUG: Computing scaling parameters from ALL data before splitting
mean = X.mean(axis=0)
std = X.std(axis=0)
X_scaled = (X - mean) / (std + 1e-8)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train logistic regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"Train Accuracy: {train_accuracy:.3f}")
print(f"Test Accuracy: {test_accuracy:.3f}")
