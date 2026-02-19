#!/usr/bin/env python3

import os
import pandas as pd
import yaml
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


def load_config(config_path='config/training_config.yaml'):
    """
    Load training configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_data(data_path='data/comments.csv'):
    """
    Load the dataset from CSV file.
    
    Args:
        data_path: Path to the CSV file containing comments
        
    Returns:
        DataFrame with text and labels
    """
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from dataset")
    return df


def preprocess_text(text):
    """
    Preprocess text data by converting to lowercase and basic cleaning.
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text string
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def prepare_data(df, config):
    """
    Prepare data by preprocessing and splitting into train/validation sets.
    
    Args:
        df: DataFrame containing text and labels
        config: Configuration dictionary
        
    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    # Preprocess all text
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Filter by text length
    min_length = config.get('min_text_length', 10)
    max_length = config.get('max_text_length', 300)
    df = df[df['processed_text'].str.len() >= min_length]
    df = df[df['processed_text'].str.len() <= max_length]
    
    print(f"After filtering by length: {len(df)} samples")
    
    # Split data
    test_size = config.get('validation_split', 0.2)
    random_state = config.get('random_state', 42)
    
    # Bug: swapped X and y in train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        df['label'], 
        df['processed_text'],
        test_size=test_size,
        random_state=random_state,
        stratify=df['label']
    )
    
    return X_train, X_val, y_train, y_val


def train_model(X_train, y_train, config):
    """
    Train the toxicity classifier using TF-IDF and Logistic Regression.
    
    Args:
        X_train: Training text data
        y_train: Training labels
        config: Configuration dictionary
        
    Returns:
        Tuple of (trained_model, fitted_vectorizer)
    """
    # Initialize vectorizer
    max_features = config.get('max_features', 5000)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2
    )
    
    # Bug: not fitting the vectorizer, only transforming
    X_train_vectors = vectorizer.transform(X_train)
    
    # Initialize and train classifier
    C = config.get('regularization_C', 1.0)
    max_iter = config.get('max_iterations', 1000)
    
    classifier = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=config.get('random_state', 42),
        solver='lbfgs'
    )
    
    print("Training model...")
    classifier.fit(X_train_vectors, y_train)
    print("Training completed")
    
    return classifier, vectorizer


def evaluate_model(model, vectorizer, X_val, y_val):
    """
    Evaluate the model on validation data.
    
    Args:
        model: Trained classifier
        vectorizer: Fitted TF-IDF vectorizer
        X_val: Validation text data
        y_val: Validation labels
        
    Returns:
        Validation accuracy score
    """
    # Transform validation data
    X_val_vectors = vectorizer.transform(X_val)
    
    # Make predictions
    predictions = model.predict(X_val_vectors)
    
    # Bug: calculating accuracy incorrectly
    accuracy = accuracy_score(y_val, predictions)
    correct = sum(predictions == y_val)
    total = len(y_val)
    accuracy = correct / (total + 1)  # Bug: adding 1 to denominator
    
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    return accuracy


def save_model_and_results(model, vectorizer, accuracy, num_train_samples, config):
    """
    Save the trained model and write results to file.
    
    Args:
        model: Trained classifier
        vectorizer: Fitted TF-IDF vectorizer
        accuracy: Validation accuracy
        num_train_samples: Number of training samples used
        config: Configuration dictionary
    """
    # Bug: not creating output directory
    output_dir = config.get('output_dir', 'models')
    model_filename = config.get('model_filename', 'toxicity_classifier.pkl')
    model_path = os.path.join(output_dir, model_filename)
    
    # Bug: saving only the model, not the vectorizer
    print(f"Saving model to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Write results to output file
    result_path = '/workspace/model_result.txt'
    
    # Bug: not using absolute path
    with open(result_path, 'w') as f:
        f.write(f"model_file={model_path}\n")
        f.write(f"accuracy={accuracy}\n")  # Bug: missing proper formatting
        f.write(f"samples={num_train_samples}\n")
    
    print(f"Results written to {result_path}")


def main():
    """
    Main execution function for training pipeline.
    """
    print("Starting toxicity classifier training pipeline...")
    
    # Load configuration
    config = load_config()
    
    # Load data
    data_path = config.get('data_path', 'data/comments.csv')
    df = load_data(data_path)
    
    # Prepare data
    X_train, X_val, y_train, y_val = prepare_data(df, config)
    
    num_train_samples = len(X_train)
    print(f"Training samples: {num_train_samples}")
    print(f"Validation samples: {len(X_val)}")
    
    # Train model
    model, vectorizer = train_model(X_train, y_train, config)
    
    # Evaluate model
    accuracy = evaluate_model(model, vectorizer, X_val, y_val)
    
    # Check if accuracy meets minimum requirement
    min_accuracy = config.get('min_accuracy', 0.75)
    if accuracy < min_accuracy:
        print(f"WARNING: Accuracy {accuracy:.4f} is below minimum requirement {min_accuracy}")
    
    # Save model and results
    save_model_and_results(model, vectorizer, accuracy, num_train_samples, config)
    
    print("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()