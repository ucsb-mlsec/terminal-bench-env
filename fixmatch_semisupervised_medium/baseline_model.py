#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# Feature columns used in the model
FEATURE_COLUMNS = ['amount', 'hour', 'day_of_week', 'merchant_category', 
                   'distance_from_home', 'distance_from_last', 'ratio_to_median']

def load_data():
    """Load all transaction datasets"""
    labeled_df = pd.read_csv('/workspace/data/labeled_transactions.csv')
    unlabeled_df = pd.read_csv('/workspace/data/unlabeled_transactions.csv')
    validation_df = pd.read_csv('/workspace/data/validation_transactions.csv')
    
    return labeled_df, unlabeled_df, validation_df

def prepare_semi_supervised_data(labeled_df, unlabeled_df):
    """
    Prepare data for semi-supervised learning by combining labeled and unlabeled data.
    Unlabeled samples are marked with -1 as the label (required by sklearn's SelfTrainingClassifier).
    """
    # Extract features and labels from labeled data
    X_labeled = labeled_df[FEATURE_COLUMNS].values
    y_labeled = labeled_df['label'].values
    
    # Extract features from unlabeled data
    X_unlabeled = unlabeled_df[FEATURE_COLUMNS].values
    y_unlabeled = np.full(len(X_unlabeled), -1)  # -1 indicates unlabeled samples
    
    # Combine labeled and unlabeled data
    X_combined = np.vstack([X_labeled, X_unlabeled])
    y_combined = np.concatenate([y_labeled, y_unlabeled])
    
    return X_combined, y_combined, len(unlabeled_df)

def train_semi_supervised_model(X_train, y_train):
    """
    Train a semi-supervised model using Self-Training approach.
    This method iteratively labels unlabeled samples with high-confidence predictions
    and adds them to the training set.
    """
    # Base classifier: Random Forest with improved parameters
    base_classifier = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    # Self-training classifier wraps the base classifier
    # threshold: minimum probability required to add pseudo-labels
    # max_iter: maximum iterations of self-training
    self_training_model = SelfTrainingClassifier(
        base_classifier,
        threshold=0.85,  # High confidence threshold for pseudo-labeling
        criterion='threshold',
        max_iter=10,
        verbose=False
    )
    
    # Fit the model on combined labeled and unlabeled data
    self_training_model.fit(X_train, y_train)
    
    return self_training_model

def evaluate_model(model, validation_df):
    """Evaluate model on validation dataset"""
    X_val = validation_df[FEATURE_COLUMNS].values
    y_val = validation_df['label'].values
    
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    
    return accuracy

def save_model(model, filepath):
    """Save trained model to disk"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")

def save_solution(model_path, accuracy, unlabeled_used):
    """Save solution results to solution.txt"""
    solution_path = '/workspace/solution.txt'
    with open(solution_path, 'w') as f:
        f.write(f"model_path={model_path}\n")
        f.write(f"validation_accuracy={accuracy:.2f}\n")
        f.write(f"unlabeled_used={unlabeled_used}\n")
    print(f"Solution saved to {solution_path}")

def main():
    print("=== Fraud Detection with Semi-Supervised Learning ===\n")
    
    # Load datasets
    print("Loading data...")
    labeled_df, unlabeled_df, validation_df = load_data()
    print(f"Labeled transactions: {len(labeled_df)}")
    print(f"Unlabeled transactions: {len(unlabeled_df)}")
    print(f"Validation transactions: {len(validation_df)}\n")
    
    # Prepare semi-supervised learning data
    print("Preparing semi-supervised learning data...")
    X_combined, y_combined, unlabeled_count = prepare_semi_supervised_data(labeled_df, unlabeled_df)
    print(f"Combined dataset size: {len(X_combined)}")
    print(f"Unlabeled samples to leverage: {unlabeled_count}\n")
    
    # Train semi-supervised model
    print("Training semi-supervised model (Self-Training)...")
    print("This approach iteratively labels high-confidence unlabeled samples")
    print("and adds them to the training set to improve the model.\n")
    model = train_semi_supervised_model(X_combined, y_combined)
    
    # Evaluate on validation set
    print("Evaluating model on validation set...")
    accuracy = evaluate_model(model, validation_df)
    print(f"Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    
    # Check if accuracy meets requirement
    if accuracy >= 0.80:
        print("✓ Success! Accuracy meets the 80% requirement.")
    else:
        print("✗ Warning: Accuracy is below 80% requirement.")
    
    # Save model
    model_path = '/workspace/fraud_model.pkl'
    save_model(model, model_path)
    
    # Save solution
    save_solution(model_path, accuracy, unlabeled_count)
    
    print("\n=== Training Complete ===")
    print(f"Model path: {model_path}")
    print(f"Validation accuracy: {accuracy:.2f}")
    print(f"Unlabeled samples used: {unlabeled_count}")

if __name__ == '__main__':
    main()