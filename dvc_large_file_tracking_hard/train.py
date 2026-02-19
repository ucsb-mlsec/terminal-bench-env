#!/usr/bin/env python3
"""
Machine Learning Model Training Pipeline

This script orchestrates the training of machine learning models with support
for configurable parameters, data preprocessing, model evaluation, and persistence.
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration YAML file
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    logger.info(f"Loading configuration from {config_path}")
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info("Configuration loaded successfully")
        logger.debug(f"Config contents: {config}")
        
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise


def load_data(csv_path: str, parquet_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training data from CSV and Parquet files.
    
    Args:
        csv_path: Path to the preprocessed CSV data file
        parquet_path: Path to the features Parquet file
        
    Returns:
        Tuple of (preprocessed_data, features) DataFrames
        
    Raises:
        FileNotFoundError: If data files don't exist
        Exception: If data loading fails
    """
    logger.info("Loading data files")
    
    # Load preprocessed CSV data
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV data file not found: {csv_path}")
    
    try:
        preprocessed_data = pd.read_csv(csv_path)
        logger.info(f"Loaded preprocessed data: {preprocessed_data.shape}")
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        raise
    
    # Load features from Parquet
    parquet_file = Path(parquet_path)
    if not parquet_file.exists():
        raise FileNotFoundError(f"Parquet features file not found: {parquet_path}")
    
    try:
        features_data = pd.read_parquet(parquet_path)
        logger.info(f"Loaded features data: {features_data.shape}")
    except Exception as e:
        logger.error(f"Error loading Parquet data: {e}")
        raise
    
    return preprocessed_data, features_data


def preprocess_features(
    data: pd.DataFrame,
    target_column: str,
    categorical_columns: list = None,
    numerical_columns: list = None,
    scale_features: bool = True
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, LabelEncoder]:
    """
    Preprocess features with scaling and encoding.
    
    Args:
        data: Input DataFrame containing features and target
        target_column: Name of the target column
        categorical_columns: List of categorical column names
        numerical_columns: List of numerical column names
        scale_features: Whether to scale numerical features
        
    Returns:
        Tuple of (X, y, scaler, label_encoder)
    """
    logger.info("Preprocessing features")
    
    if categorical_columns is None:
        categorical_columns = []
    if numerical_columns is None:
        numerical_columns = []
    
    # Separate features and target
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    y = data[target_column].values
    X = data.drop(columns=[target_column])
    
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    # Handle categorical features
    if categorical_columns:
        logger.info(f"Encoding {len(categorical_columns)} categorical columns")
        for col in categorical_columns:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle numerical features
    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    
    if scale_features and numerical_columns:
        logger.info(f"Scaling {len(numerical_columns)} numerical columns")
        X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
    elif scale_features:
        logger.info("Scaling all numerical features")
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    # Encode target if it's categorical
    if y.dtype == 'object' or y.dtype.name == 'category':
        logger.info("Encoding target variable")
        y = label_encoder.fit_transform(y)
    
    logger.info("Feature preprocessing completed")
    
    return X.values, y, scaler, label_encoder


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict[str, Any]
) -> RandomForestClassifier:
    """
    Train RandomForest classifier with configured parameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        config: Configuration dictionary with model parameters
        
    Returns:
        Trained RandomForestClassifier model
    """
    logger.info("Training RandomForest model")
    
    # Extract model parameters from config
    model_params = config.get('model_parameters', {})
    
    n_estimators = model_params.get('n_estimators', 100)
    max_depth = model_params.get('max_depth', None)
    min_samples_split = model_params.get('min_samples_split', 2)
    min_samples_leaf = model_params.get('min_samples_leaf', 1)
    random_state = model_params.get('random_state', 42)
    n_jobs = model_params.get('n_jobs', -1)
    
    logger.info(f"Model parameters: n_estimators={n_estimators}, max_depth={max_depth}")
    
    # Initialize and train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1
    )
    
    logger.info(f"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features")
    model.fit(X_train, y_train)
    
    logger.info("Model training completed")
    
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary containing evaluation metrics
    """
    logger.info("Evaluating model performance")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    
    logger.info(f"Model Evaluation Results:")
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1-Score:  {f1:.4f}")
    
    return metrics


def save_model(
    model: RandomForestClassifier,
    scaler: StandardScaler,
    label_encoder: LabelEncoder,
    metrics: Dict[str, float],
    output_path: str,
    model_name: str = "random_forest_model"
) -> None:
    """
    Save trained model and associated artifacts to disk.
    
    Args:
        model: Trained model to save
        scaler: Feature scaler
        label_encoder: Target encoder
        metrics: Model evaluation metrics
        output_path: Directory path to save model
        model_name: Base name for model files
    """
    logger.info(f"Saving model to {output_path}")
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_file = output_dir / f"{model_name}.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_file}")
    
    # Save scaler
    scaler_file = output_dir / f"{model_name}_scaler.pkl"
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {scaler_file}")
    
    # Save label encoder
    encoder_file = output_dir / f"{model_name}_encoder.pkl"
    with open(encoder_file, 'wb') as f:
        pickle.dump(label_encoder, f)
    logger.info(f"Label encoder saved to {encoder_file}")
    
    # Save metrics
    metrics_file = output_dir / f"{model_name}_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_file}")
    
    logger.info("All model artifacts saved successfully")


def main():
    """
    Main orchestration function for the training pipeline.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train machine learning model with configurable parameters"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/model_config.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data',
        help='Path to data directory'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='models',
        help='Path to output directory for saved models'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Construct data file paths
        csv_path = Path(args.data_path) / 'preprocessed_data.csv'
        parquet_path = Path(args.data_path) / 'features.parquet'
        
        # Load data
        preprocessed_data, features_data = load_data(str(csv_path), str(parquet_path))
        
        # Merge data if needed (use features_data as primary)
        if 'target' in preprocessed_data.columns and 'target' not in features_data.columns:
            data = features_data.copy()
            data['target'] = preprocessed_data['target']
        else:
            data = features_data
        
        # Get preprocessing parameters from config
        preprocess_config = config.get('preprocessing', {})
        target_column = preprocess_config.get('target_column', 'target')
        categorical_columns = preprocess_config.get('categorical_columns', [])
        numerical_columns = preprocess_config.get('numerical_columns', [])
        
        # Preprocess features
        X, y, scaler, label_encoder = preprocess_features(
            data=data,
            target_column=target_column,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            scale_features=True
        )
        
        # Split data
        test_size = config.get('training', {}).get('test_size', 0.2)
        random_state = config.get('model_parameters', {}).get('random_state', 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Train model
        model = train_model(X_train, y_train, config)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save model and artifacts
        model_name = config.get('model_name', 'random_forest_model')
        save_model(
            model=model,
            scaler=scaler,
            label_encoder=label_encoder,
            metrics=metrics,
            output_path=args.output_path,
            model_name=model_name
        )
        
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()