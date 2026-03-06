#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score
from collections import defaultdict

def split_train_test(data, test_ratio=0.2):
    """
    Split data into train/test sets.
    BUG: Not stratified properly - uses random split instead of per-class split
    """
    n_total = len(data)
    n_test = int(n_total * test_ratio)
    
    # BUG: Should use stratified split to ensure all classes are in both sets
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    train_data = data.iloc[train_indices].reset_index(drop=True)
    test_data = data.iloc[test_indices].reset_index(drop=True)
    
    return train_data, test_data

def calculate_accuracy(predictions, labels):
    """
    Calculate classification accuracy.
    BUG: Wrong comparison logic - compares strings with integers incorrectly
    """
    if len(predictions) == 0:
        return 0.0
    
    # BUG: Should convert both to same type before comparison
    correct = 0
    for pred, label in zip(predictions, labels):
        if pred == label:
            correct += 1
    
    # BUG: Integer division in Python 2 style (doesn't affect Python 3 but logic error)
    accuracy = correct / len(predictions)
    return accuracy

def save_model(model_state, filepath):
    """
    Save model state to file using pickle.
    BUG: Doesn't create directory if it doesn't exist
    """
    # BUG: Should create parent directories first
    # os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_state, f)
    
    return True

def normalize_embeddings(embeddings):
    """
    Normalize embedding vectors to unit length.
    BUG: Wrong axis for norm calculation
    """
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)
    
    # BUG: Using wrong axis - should be axis=1 for row-wise norm
    norms = np.linalg.norm(embeddings, axis=0, keepdims=True)
    
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    
    normalized = embeddings / norms
    return normalized

def sample_balanced(data, n_samples_per_class):
    """
    Sample equal number of examples from each class.
    BUG: Doesn't handle case when class has fewer samples than requested
    """
    categories = data['category'].unique()
    sampled_data = []
    
    for category in categories:
        class_data = data[data['category'] == category]
        
        # BUG: Should check if len(class_data) >= n_samples_per_class
        # and handle the case appropriately (sample with replacement or raise error)
        sampled = class_data.sample(n=n_samples_per_class, replace=False)
        sampled_data.append(sampled)
    
    result = pd.concat(sampled_data, ignore_index=True)
    return result

def euclidean_distance(x, y):
    """
    Calculate Euclidean distance between two vectors or matrices.
    BUG: Incorrect broadcasting for batch computation
    """
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    if len(y.shape) == 1:
        y = y.reshape(1, -1)
    
    # BUG: Wrong computation - should use proper broadcasting
    # Correct: np.sqrt(np.sum((x[:, np.newaxis, :] - y[np.newaxis, :, :]) ** 2, axis=2))
    diff = x - y
    distances = np.sqrt(np.sum(diff ** 2, axis=1))
    
    return distances

def create_episode(data, n_way, n_support, n_query):
    """
    Create an episode for few-shot learning.
    BUG: Samples same examples for support and query sets
    """
    categories = data['category'].unique()
    
    # BUG: Should ensure n_way <= len(categories)
    selected_classes = np.random.choice(categories, size=n_way, replace=False)
    
    support_set = []
    query_set = []
    
    for cls in selected_classes:
        class_data = data[data['category'] == cls]
        
        # BUG: Should sample without replacement and ensure support/query are disjoint
        support_samples = class_data.sample(n=n_support, replace=False)
        query_samples = class_data.sample(n=n_query, replace=False)
        
        support_set.append(support_samples)
        query_set.append(query_samples)
    
    support_df = pd.concat(support_set, ignore_index=True)
    query_df = pd.concat(query_set, ignore_index=True)
    
    return support_df, query_df, selected_classes

def compute_prototypes(embeddings, labels, classes):
    """
    Compute class prototypes from support embeddings.
    BUG: Wrong averaging logic
    """
    prototypes = {}
    
    for cls in classes:
        # BUG: Should use proper indexing
        class_mask = labels == cls
        class_embeddings = embeddings[class_mask]
        
        # BUG: Wrong axis for mean
        prototype = np.mean(class_embeddings, axis=1)
        prototypes[cls] = prototype
    
    return prototypes