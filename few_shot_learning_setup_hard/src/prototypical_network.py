#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import random
import pickle
import os
from collections import defaultdict

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

def load_data(filepath):
    """
    Load CSV data from filepath.
    BUG: Uses wrong column name 'sentiment' instead of 'category'
    """
    df = pd.read_csv(filepath)
    # BUG: Looking for 'sentiment' column instead of 'category'
    texts = df['text'].tolist()
    labels = df['sentiment'].tolist()  # Should be 'category'
    return texts, labels

def create_episode(data, n_way, n_support, n_query):
    """
    Create an episode by sampling classes and examples.
    BUG: Incorrect sampling logic that may sample same indices multiple times
    """
    texts, labels = data
    unique_labels = list(set(labels))
    
    # Sample n_way classes
    episode_classes = random.sample(unique_labels, n_way)
    
    support_texts = []
    support_labels = []
    query_texts = []
    query_labels = []
    
    for class_idx, class_label in enumerate(episode_classes):
        # Get all indices for this class
        class_indices = [i for i, label in enumerate(labels) if label == class_label]
        
        # BUG: Using random.choice with replacement, may select duplicates
        support_indices = [random.choice(class_indices) for _ in range(n_support)]
        query_indices = [random.choice(class_indices) for _ in range(n_query)]
        
        for idx in support_indices:
            support_texts.append(texts[idx])
            support_labels.append(class_idx)
        
        for idx in query_indices:
            query_texts.append(texts[idx])
            query_labels.append(class_idx)
    
    return support_texts, support_labels, query_texts, query_labels, episode_classes

def generate_embeddings(texts, vectorizer=None, fit=False):
    """
    Generate text embeddings using TfidfVectorizer.
    BUG: Inconsistent handling of vectorizer fitting
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=500)
        fit = True
    
    if fit:
        # BUG: Always fitting even when vectorizer is provided
        embeddings = vectorizer.fit_transform(texts).toarray()
    else:
        embeddings = vectorizer.transform(texts).toarray()
    
    # BUG: Missing normalization that causes distance computation issues
    return embeddings, vectorizer

def compute_prototypes(support_embeddings, support_labels):
    """
    Compute class prototypes by averaging support embeddings.
    BUG: Wrong axis for mean calculation
    """
    unique_labels = sorted(set(support_labels))
    prototypes = []
    
    for label in unique_labels:
        label_indices = [i for i, l in enumerate(support_labels) if l == label]
        label_embeddings = support_embeddings[label_indices]
        # BUG: Using axis=1 instead of axis=0, causing wrong dimension
        prototype = np.mean(label_embeddings, axis=1)
        prototypes.append(prototype)
    
    return np.array(prototypes)

def compute_distances(query_embeddings, prototypes):
    """
    Compute euclidean distances between queries and prototypes.
    BUG: Wrong distance calculation formula
    """
    n_queries = query_embeddings.shape[0]
    n_prototypes = prototypes.shape[0]
    distances = np.zeros((n_queries, n_prototypes))
    
    for i in range(n_queries):
        for j in range(n_prototypes):
            # BUG: Using sum instead of proper euclidean distance
            distances[i, j] = np.sum((query_embeddings[i] - prototypes[j]))
    
    return distances

def classify(distances):
    """
    Classify based on distances to prototypes.
    BUG: Using argmax instead of argmin
    """
    # BUG: Should use argmin for distances, but using argmax
    predictions = np.argmax(distances, axis=1)
    return predictions

def train_episode(data, n_way, n_support, n_query, vectorizer):
    """
    Run one training episode.
    BUG: Doesn't properly update or return useful training signals
    """
    support_texts, support_labels, query_texts, query_labels, episode_classes = create_episode(
        data, n_way, n_support, n_query
    )
    
    # Generate embeddings
    support_embeddings, _ = generate_embeddings(support_texts, vectorizer, fit=False)
    query_embeddings, _ = generate_embeddings(query_texts, vectorizer, fit=False)
    
    # Compute prototypes
    prototypes = compute_prototypes(support_embeddings, support_labels)
    
    # Compute distances
    distances = compute_distances(query_embeddings, prototypes)
    
    # Classify
    predictions = classify(distances)
    
    # Calculate accuracy
    accuracy = np.mean(np.array(predictions) == np.array(query_labels))
    
    # BUG: Not returning any gradient or update mechanism
    return accuracy

def test(data, n_way, n_support, n_query, n_episodes, vectorizer):
    """
    Test the model over multiple episodes.
    """
    accuracies = []
    
    for episode in range(n_episodes):
        support_texts, support_labels, query_texts, query_labels, episode_classes = create_episode(
            data, n_way, n_support, n_query
        )
        
        support_embeddings, _ = generate_embeddings(support_texts, vectorizer, fit=False)
        query_embeddings, _ = generate_embeddings(query_texts, vectorizer, fit=False)
        
        prototypes = compute_prototypes(support_embeddings, support_labels)
        distances = compute_distances(query_embeddings, prototypes)
        predictions = classify(distances)
        
        accuracy = np.mean(np.array(predictions) == np.array(query_labels))
        accuracies.append(accuracy)
    
    return np.mean(accuracies)

def main():
    """
    Main function to run training and testing.
    """
    # Load data
    data_path = '/workspace/data/feedback.csv'
    texts, labels = load_data(data_path)
    
    # Split data into train and test
    indices = list(range(len(texts)))
    random.shuffle(indices)
    split_idx = int(0.7 * len(indices))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_texts = [texts[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_texts = [texts[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    train_data = (train_texts, train_labels)
    test_data = (test_texts, test_labels)
    
    # Create and fit vectorizer on all training data
    vectorizer = TfidfVectorizer(max_features=500)
    vectorizer.fit(train_texts)
    
    # Training parameters
    n_way = 5
    n_support = 5
    n_query = 5
    n_train_episodes = 100
    n_test_episodes = 50
    
    print("Starting training...")
    train_accuracies = []
    for episode in range(n_train_episodes):
        acc = train_episode(train_data, n_way, n_support, n_query, vectorizer)
        train_accuracies.append(acc)
        if (episode + 1) % 20 == 0:
            print(f"Episode {episode + 1}/{n_train_episodes}, Accuracy: {acc:.4f}")
    
    print("\nStarting testing...")
    test_accuracy = test(test_data, n_way, n_support, n_query, n_test_episodes, vectorizer)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save model
    model_data = {
        'vectorizer': vectorizer,
        'n_way': n_way,
        'n_support': n_support
    }
    
    os.makedirs('/workspace/models', exist_ok=True)
    with open('/workspace/models/prototypical_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    # Save results
    with open('/workspace/result.txt', 'w') as f:
        f.write(f"test_accuracy={test_accuracy:.2f}\n")
        f.write("model_saved=yes\n")
    
    print("\nResults saved to /workspace/result.txt")

if __name__ == "__main__":
    main()