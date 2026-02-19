#!/usr/bin/env python3
"""
Training script for transaction data machine learning model.
This script loads transaction data from multiple CSV files,
preprocesses the data, and trains a classifier.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import os

def load_data():
    """
    Load transaction data from multiple CSV files.
    
    Returns:
        pd.DataFrame: Combined dataframe with all transaction data
    """
    print("Loading transaction data...")
    
    # Define paths to data files
    data_files = [
        'data/raw/transactions_2023.csv',
        'data/raw/transactions_2024_q1.csv',
        'data/raw/transactions_2024_q2.csv'
    ]
    
    # Read and concatenate all CSV files
    dataframes = []
    for file_path in data_files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            dataframes.append(df)
            print(f"Loaded {file_path}: {len(df)} rows")
        else:
            print(f"Warning: {file_path} not found")
    
    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Total rows loaded: {len(combined_df)}")
    
    return combined_df

def train_model(data):
    """
    Train a machine learning model on transaction data.
    
    Args:
        data (pd.DataFrame): Transaction data
    
    Returns:
        tuple: Trained model and accuracy score
    """
    print("\nPreprocessing data...")
    
    # Assume the dataframe has columns like 'amount', 'category', 'status'
    # For this example, we'll create sample preprocessing
    
    # Handle missing values
    data = data.fillna(0)
    
    # Encode categorical variables
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        if column != 'status':  # Assuming 'status' is the target variable
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column].astype(str))
            label_encoders[column] = le
    
    # Separate features and target
    # Assuming 'status' is the target variable
    if 'status' in data.columns:
        X = data.drop('status', axis=1)
        y = data['status']
        
        # Encode target variable if it's categorical
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
    else:
        # If no status column, create dummy target
        print("No 'status' column found, creating dummy target")
        X = data
        y = np.random.randint(0, 2, len(data))
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train Random Forest Classifier
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate and print accuracy metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, accuracy

def main():
    """
    Main function to orchestrate data loading and model training.
    """
    print("="*60)
    print("Transaction Data ML Training Pipeline")
    print("="*60)
    
    # Load data from CSV files
    data = load_data()
    
    # Train the model
    model, accuracy = train_model(data)
    
    print("\n" + "="*60)
    print(f"Training completed successfully!")
    print(f"Final Model Accuracy: {accuracy:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()