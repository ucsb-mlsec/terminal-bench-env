#!/usr/bin/env python3
"""
Memory-inefficient training script that will crash with OOM errors.
This implementation has multiple memory issues that need to be resolved.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import psutil
import os
import tracemalloc

# Start memory tracking
tracemalloc.start()
process = psutil.Process(os.getpid())

def get_memory_usage():
    """Get current memory usage in MB"""
    return process.memory_info().rss / 1024 / 1024

print("Starting training script...")
print(f"Initial memory usage: {get_memory_usage():.2f} MB")

# MEMORY ISSUE 1: Loading entire CSV without optimization
# No dtype specification, no chunksize, loads everything into memory at once
print("Loading data...")
df = pd.read_csv('data/training_data.csv')
print(f"Data loaded. Shape: {df.shape}")
print(f"Memory after loading: {get_memory_usage():.2f} MB")

# MEMORY ISSUE 2: Creating unnecessary full copies of dataframe
print("Creating backup copies...")
df_backup = df.copy()  # Unnecessary full copy
df_original = df.copy()  # Another unnecessary copy
df_working = df.copy()  # Yet another copy
print(f"Memory after copies: {get_memory_usage():.2f} MB")

# MEMORY ISSUE 3: More unnecessary copies during preprocessing
print("Preprocessing data...")
df_copy1 = df.copy()
df_copy2 = df.copy()

# Simulate target variable (last column as target)
X = df.iloc[:, :-1].copy()  # Another copy
y = df.iloc[:, -1].copy()  # Another copy

# MEMORY ISSUE 4: Creating multiple intermediate dataframes without cleanup
print("Feature engineering...")
# Creating scaled version without freeing original
X_for_scaling = X.copy()
scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(X_for_scaling)
X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns)  # Convert back to DataFrame

# MEMORY ISSUE 5: One-hot encoding creates massive expansion
print("One-hot encoding...")
X_encoded = pd.get_dummies(X_scaled, columns=X_scaled.columns[:10])  # Encoding creates many columns
print(f"After encoding shape: {X_encoded.shape}")
print(f"Memory after encoding: {get_memory_usage():.2f} MB")

# MEMORY ISSUE 6: Multiple numpy array conversions without cleanup
print("Converting to numpy arrays...")
X_array1 = X_encoded.values.copy()
X_array2 = X_encoded.values.copy()
X_array3 = X_encoded.values.copy()
y_array1 = y.values.copy()
y_array2 = y.values.copy()

print(f"Memory after array conversions: {get_memory_usage():.2f} MB")

# Keep all the intermediate data in memory
all_copies = [df_backup, df_original, df_working, df_copy1, df_copy2, 
              X, X_for_scaling, X_scaled, X_encoded]

# Train test split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_array1, y_array1, test_size=0.2, random_state=42
)

# More unnecessary copies
X_train_copy = X_train.copy()
X_test_copy = X_test.copy()
y_train_copy = y_train.copy()
y_test_copy = y_test.copy()

print(f"Memory before training: {get_memory_usage():.2f} MB")

# MEMORY ISSUE 7: Using memory-intensive Random Forest with default parameters
print("Training model...")
model = RandomForestClassifier(
    n_estimators=200,  # High number of trees
    max_depth=None,  # No depth limit - very memory intensive
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1  # Use all cores, more memory
)

# Train on the copied data
model.fit(X_train_copy, y_train_copy)

print("Training completed!")
print(f"Model score: {model.score(X_test_copy, y_test_copy):.4f}")

# Get peak memory usage
current_memory = get_memory_usage()
peak_memory = tracemalloc.get_traced_memory()[1] / 1024 / 1024

print(f"\nFinal memory usage: {current_memory:.2f} MB")
print(f"Peak memory usage: {peak_memory:.2f} MB")

# Save results
rows_processed = len(df)
training_completed = "yes"

with open('/workspace/solution.txt', 'w') as f:
    f.write(f"peak_memory_mb={int(current_memory)}\n")
    f.write(f"training_completed={training_completed}\n")
    f.write(f"rows_processed={rows_processed}\n")

print(f"\nResults saved to solution.txt")
print(f"Rows processed: {rows_processed}")

tracemalloc.stop()