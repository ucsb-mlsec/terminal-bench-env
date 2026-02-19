#!/usr/bin/env python3
"""
Text Classification System
==========================

This script implements a simple text classifier with a frozen feature extractor
and tunable parameters for the decision boundary.

Usage:
    # Load parameters and classify text
    params = load_parameters('current_params.txt')
    prediction = classify_text("This is great!", params)
    
    # Evaluate on a dataset
    accuracy = evaluate('val.jsonl', params)

The system works as follows:
1. Feature extractor converts text to 5 numerical features (FROZEN - cannot modify)
2. Parameters (5 floats) are combined with features using dot product
3. Classification threshold determines final prediction (0 or 1)
"""

import json
import sys
import math


def load_parameters(filepath):
    """
    Load 5 floating-point parameters from a text file.
    
    Args:
        filepath: Path to parameter file (5 lines, one float per line)
    
    Returns:
        List of 5 floats
    """
    with open(filepath, 'r') as f:
        params = [float(line.strip()) for line in f.readlines()]
    
    if len(params) != 5:
        raise ValueError(f"Expected 5 parameters, got {len(params)}")
    
    return params


def extract_features(text):
    """
    FROZEN FEATURE EXTRACTOR - DO NOT MODIFY
    
    Extracts 5 numerical features from input text.
    This simulates a pre-trained feature extraction model.
    
    Features:
    1. Text length (normalized by 100)
    2. Word count (normalized by 20)
    3. Punctuation ratio
    4. Average word length (normalized by 10)
    5. Positive sentiment indicator (presence of positive words)
    
    Args:
        text: Input text string
    
    Returns:
        List of 5 floats representing features
    """
    # Feature 1: Text length (normalized)
    text_length = len(text) / 100.0
    
    # Feature 2: Word count (normalized)
    words = text.split()
    word_count = len(words) / 20.0
    
    # Feature 3: Punctuation ratio
    punctuation = "!?.,;:"
    punct_count = sum(1 for char in text if char in punctuation)
    punct_ratio = punct_count / max(len(text), 1)
    
    # Feature 4: Average word length (normalized)
    if words:
        avg_word_length = sum(len(word) for word in words) / len(words) / 10.0
    else:
        avg_word_length = 0.0
    
    # Feature 5: Positive sentiment indicator
    positive_words = {
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
        'love', 'best', 'perfect', 'awesome', 'happy', 'pleased', 'satisfied',
        'thank', 'thanks', 'appreciate', 'recommend', 'beautiful', 'brilliant',
        'outstanding', 'superb', 'delighted', 'exceptional', 'fabulous'
    }
    negative_words = {
        'bad', 'terrible', 'horrible', 'awful', 'worst', 'hate', 'poor',
        'disappointing', 'disappointed', 'waste', 'useless', 'never', 'not',
        'broken', 'defective', 'unhappy', 'unsatisfied', 'annoyed', 'angry',
        'frustrating', 'frustrated', 'pathetic', 'disgusting', 'refund'
    }
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    # Sentiment score: positive contribution minus negative contribution
    sentiment_score = (pos_count - neg_count) / 5.0
    
    features = [
        text_length,
        word_count,
        punct_ratio,
        avg_word_length,
        sentiment_score
    ]
    
    return features


def compute_score(features, parameters):
    """
    Combine features with parameters using dot product.
    
    Args:
        features: List of 5 floats (feature values)
        parameters: List of 5 floats (tunable parameters)
    
    Returns:
        Float score
    """
    if len(features) != 5 or len(parameters) != 5:
        raise ValueError("Features and parameters must both have length 5")
    
    # Dot product
    score = sum(f * p for f, p in zip(features, parameters))
    return score


def classify_text(text, parameters, threshold=0.0):
    """
    Classify text as positive (1) or negative (0) sentiment.
    
    Args:
        text: Input text string
        parameters: List of 5 floats (tunable parameters)
        threshold: Decision threshold (default 0.0)
    
    Returns:
        Integer: 0 (negative) or 1 (positive)
    """
    features = extract_features(text)
    score = compute_score(features, parameters)
    
    # Decision: if score > threshold, predict positive (1), else negative (0)
    prediction = 1 if score > threshold else 0
    return prediction


def load_jsonl_data(filepath):
    """
    Load data from JSONL file.
    
    Args:
        filepath: Path to JSONL file
    
    Returns:
        List of dictionaries with 'text' and 'label' keys
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def evaluate(data_filepath, parameters):
    """
    Evaluate classifier accuracy on a dataset.
    
    Args:
        data_filepath: Path to JSONL data file
        parameters: List of 5 floats (tunable parameters)
    
    Returns:
        Float: Accuracy (between 0.0 and 1.0)
    """
    data = load_jsonl_data(data_filepath)
    
    if not data:
        return 0.0
    
    correct = 0
    total = len(data)
    
    for sample in data:
        text = sample['text']
        true_label = sample['label']
        predicted_label = classify_text(text, parameters)
        
        if predicted_label == true_label:
            correct += 1
    
    accuracy = correct / total
    return accuracy


def main():
    """
    Example usage of the classifier.
    """
    # Example: Load parameters and evaluate
    if len(sys.argv) > 1:
        param_file = sys.argv[1]
    else:
        param_file = '/workspace/current_params.txt'
    
    print(f"Loading parameters from: {param_file}")
    params = load_parameters(param_file)
    print(f"Parameters: {params}")
    
    # Test on validation data
    val_file = '/workspace/data/val.jsonl'
    print(f"\nEvaluating on: {val_file}")
    accuracy = evaluate(val_file, params)
    print(f"Accuracy: {accuracy:.2%}")
    
    # Example classification
    example_texts = [
        "This product is amazing! I love it!",
        "Terrible quality. Complete waste of money."
    ]
    
    print("\nExample classifications:")
    for text in example_texts:
        pred = classify_text(text, params)
        sentiment = "Positive" if pred == 1 else "Negative"
        print(f"  '{text}' -> {sentiment} ({pred})")


if __name__ == "__main__":
    main()