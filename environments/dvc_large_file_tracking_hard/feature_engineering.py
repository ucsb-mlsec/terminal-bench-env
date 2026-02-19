#!/usr/bin/env python3
"""
Feature Engineering Pipeline
Transforms preprocessed customer data into ML-ready features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pyarrow.parquet as pq
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_preprocessed():
    """
    Load preprocessed customer data from CSV file
    
    Returns:
        pd.DataFrame: Preprocessed customer data
    """
    logger.info("Loading preprocessed data...")
    try:
        df = pd.read_csv('data/preprocessed_data.csv')
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        logger.error("Preprocessed data file not found at data/preprocessed_data.csv")
        raise
    except Exception as e:
        logger.error(f"Error loading preprocessed data: {str(e)}")
        raise


def calculate_recency(df, reference_date='2024-01-01'):
    """
    Calculate recency score - days since last purchase
    
    Args:
        df (pd.DataFrame): Customer dataframe with last_purchase_date column
        reference_date (str): Reference date for recency calculation
        
    Returns:
        pd.Series: Recency in days
    """
    logger.info("Calculating recency scores...")
    try:
        ref_date = pd.to_datetime(reference_date)
        df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
        recency = (ref_date - df['last_purchase_date']).dt.days
        logger.info(f"Recency calculated - min: {recency.min()}, max: {recency.max()}, mean: {recency.mean():.2f}")
        return recency
    except Exception as e:
        logger.error(f"Error calculating recency: {str(e)}")
        raise


def calculate_frequency_score(df):
    """
    Calculate frequency score - normalized total transactions (0-10 scale)
    
    Args:
        df (pd.DataFrame): Customer dataframe with total_transactions column
        
    Returns:
        pd.Series: Normalized frequency score (0-10)
    """
    logger.info("Calculating frequency scores...")
    try:
        transactions = df['total_transactions']
        min_val = transactions.min()
        max_val = transactions.max()
        
        if max_val == min_val:
            frequency_score = pd.Series([5.0] * len(df), index=df.index)
        else:
            frequency_score = 10 * (transactions - min_val) / (max_val - min_val)
        
        logger.info(f"Frequency score calculated - min: {frequency_score.min():.2f}, max: {frequency_score.max():.2f}")
        return frequency_score
    except Exception as e:
        logger.error(f"Error calculating frequency score: {str(e)}")
        raise


def calculate_monetary_score(df):
    """
    Calculate monetary score - normalized lifetime value (0-10 scale)
    
    Args:
        df (pd.DataFrame): Customer dataframe with lifetime_value column
        
    Returns:
        pd.Series: Normalized monetary score (0-10)
    """
    logger.info("Calculating monetary scores...")
    try:
        lifetime_value = df['lifetime_value']
        min_val = lifetime_value.min()
        max_val = lifetime_value.max()
        
        if max_val == min_val:
            monetary_score = pd.Series([5.0] * len(df), index=df.index)
        else:
            monetary_score = 10 * (lifetime_value - min_val) / (max_val - min_val)
        
        logger.info(f"Monetary score calculated - min: {monetary_score.min():.2f}, max: {monetary_score.max():.2f}")
        return monetary_score
    except Exception as e:
        logger.error(f"Error calculating monetary score: {str(e)}")
        raise


def encode_categories(df):
    """
    Encode preferred_category to integer values
    
    Mapping:
    - Electronics: 0
    - Clothing: 1
    - Food: 2
    - Home: 3
    - Sports: 4
    - Books: 5
    - Toys: 6
    
    Args:
        df (pd.DataFrame): Customer dataframe with preferred_category column
        
    Returns:
        pd.Series: Encoded category values
    """
    logger.info("Encoding category features...")
    try:
        category_mapping = {
            'Electronics': 0,
            'Clothing': 1,
            'Food': 2,
            'Home': 3,
            'Sports': 4,
            'Books': 5,
            'Toys': 6
        }
        
        encoded = df['preferred_category'].map(category_mapping)
        
        # Handle any unmapped categories
        if encoded.isna().any():
            logger.warning(f"Found {encoded.isna().sum()} unmapped categories, filling with -1")
            encoded = encoded.fillna(-1).astype(int)
        else:
            encoded = encoded.astype(int)
        
        logger.info(f"Categories encoded - unique values: {encoded.unique()}")
        return encoded
    except Exception as e:
        logger.error(f"Error encoding categories: {str(e)}")
        raise


def calculate_derived_features(df):
    """
    Calculate derived features from existing columns
    
    Features:
    - avg_basket_size: lifetime_value / total_transactions
    - payment_diversity: placeholder value (0.5)
    - seasonal_pattern: sine wave based on last_purchase month
    - churn_risk: calculated from recency and frequency
    
    Args:
        df (pd.DataFrame): Customer dataframe with required columns
        
    Returns:
        pd.DataFrame: DataFrame with derived features
    """
    logger.info("Calculating derived features...")
    try:
        derived = pd.DataFrame(index=df.index)
        
        # Average basket size
        derived['avg_basket_size'] = df['lifetime_value'] / df['total_transactions'].replace(0, 1)
        logger.info(f"Average basket size - mean: {derived['avg_basket_size'].mean():.2f}")
        
        # Payment diversity (placeholder)
        derived['payment_diversity'] = 0.5
        logger.info("Payment diversity set to placeholder value 0.5")
        
        # Seasonal pattern based on last purchase month
        df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
        month = df['last_purchase_date'].dt.month
        derived['seasonal_pattern'] = np.sin(2 * np.pi * month / 12)
        logger.info(f"Seasonal pattern - min: {derived['seasonal_pattern'].min():.2f}, max: {derived['seasonal_pattern'].max():.2f}")
        
        # Churn risk calculation
        # Based on recency (higher = more risk) and frequency (higher = less risk)
        recency = calculate_recency(df)
        frequency_score = calculate_frequency_score(df)
        
        # Normalize recency to 0-1 (inverted so higher = more risk)
        recency_normalized = (recency - recency.min()) / (recency.max() - recency.min() + 1e-10)
        frequency_normalized = frequency_score / 10.0
        
        # Churn risk: high recency + low frequency = high risk
        derived['churn_risk'] = (recency_normalized * 0.6 + (1 - frequency_normalized) * 0.4)
        logger.info(f"Churn risk - min: {derived['churn_risk'].min():.2f}, max: {derived['churn_risk'].max():.2f}, mean: {derived['churn_risk'].mean():.2f}")
        
        return derived
    except Exception as e:
        logger.error(f"Error calculating derived features: {str(e)}")
        raise


def assign_clusters(df):
    """
    Assign customer clusters using KMeans clustering
    
    Uses 5 clusters based on normalized features:
    - recency, frequency_score, monetary_score, avg_basket_size, churn_risk
    
    Args:
        df (pd.DataFrame): Customer dataframe with all features
        
    Returns:
        pd.Series: Cluster assignments (0-4)
    """
    logger.info("Assigning customer clusters...")
    try:
        # Select features for clustering
        feature_cols = ['recency', 'frequency_score', 'monetary_score', 
                       'avg_basket_size', 'churn_risk']
        
        # Check if all features exist
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns for clustering: {missing_cols}")
            raise ValueError(f"Missing columns: {missing_cols}")
        
        clustering_data = df[feature_cols].copy()
        
        # Handle any missing values
        if clustering_data.isna().any().any():
            logger.warning("Found NaN values in clustering features, filling with column means")
            clustering_data = clustering_data.fillna(clustering_data.mean())
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(clustering_data)
        
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_features)
        
        # Log cluster distribution
        unique, counts = np.unique(clusters, return_counts=True)
        cluster_dist = dict(zip(unique, counts))
        logger.info(f"Cluster distribution: {cluster_dist}")
        
        return pd.Series(clusters, index=df.index, name='cluster')
    except Exception as e:
        logger.error(f"Error assigning clusters: {str(e)}")
        raise


def save_features(df, output_path='data/features.parquet'):
    """
    Save engineered features to Parquet file
    
    Args:
        df (pd.DataFrame): Feature dataframe to save
        output_path (str): Output file path
    """
    logger.info(f"Saving features to {output_path}...")
    try:
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to parquet
        df.to_parquet(output_path, index=False, compression='snappy')
        
        # Verify file was created
        file_size = os.path.getsize(output_path)
        logger.info(f"Features saved successfully - {len(df)} records, {file_size:,} bytes")
    except Exception as e:
        logger.error(f"Error saving features: {str(e)}")
        raise


def main():
    """
    Main orchestration function for feature engineering pipeline
    """
    logger.info("=" * 60)
    logger.info("Starting Feature Engineering Pipeline")
    logger.info("=" * 60)
    
    try:
        # Step 1: Load preprocessed data
        df = load_preprocessed()
        
        # Step 2: Calculate RFM features
        df['recency'] = calculate_recency(df)
        df['frequency_score'] = calculate_frequency_score(df)
        df['monetary_score'] = calculate_monetary_score(df)
        
        # Step 3: Encode categorical features
        df['category_encoded'] = encode_categories(df)
        
        # Step 4: Calculate derived features
        derived_features = calculate_derived_features(df)
        df = pd.concat([df, derived_features], axis=1)
        
        # Step 5: Assign clusters
        df['cluster'] = assign_clusters(df)
        
        # Step 6: Select final feature set
        feature_columns = [
            'customer_id',
            'recency',
            'frequency_score',
            'monetary_score',
            'category_encoded',
            'avg_basket_size',
            'payment_diversity',
            'seasonal_pattern',
            'churn_risk',
            'cluster'
        ]
        
        # Include original columns that might be needed
        original_columns = ['total_transactions', 'lifetime_value', 
                          'preferred_category', 'last_purchase_date']
        for col in original_columns:
            if col in df.columns and col not in feature_columns:
                feature_columns.append(col)
        
        # Create final feature dataframe
        features_df = df[feature_columns].copy()
        
        logger.info(f"Final feature set: {len(features_df)} records, {len(features_df.columns)} features")
        logger.info(f"Features: {list(features_df.columns)}")
        
        # Step 7: Save features
        save_features(features_df)
        
        # Summary statistics
        logger.info("=" * 60)
        logger.info("Feature Engineering Summary")
        logger.info("=" * 60)
        logger.info(f"Total records processed: {len(features_df)}")
        logger.info(f"Total features created: {len(features_df.columns)}")
        logger.info(f"Recency range: {features_df['recency'].min():.0f} - {features_df['recency'].max():.0f} days")
        logger.info(f"Frequency score range: {features_df['frequency_score'].min():.2f} - {features_df['frequency_score'].max():.2f}")
        logger.info(f"Monetary score range: {features_df['monetary_score'].min():.2f} - {features_df['monetary_score'].max():.2f}")
        logger.info(f"Average churn risk: {features_df['churn_risk'].mean():.2f}")
        logger.info(f"Number of clusters: {features_df['cluster'].nunique()}")
        logger.info("=" * 60)
        logger.info("Feature Engineering Pipeline Completed Successfully")
        logger.info("=" * 60)
        
        return features_df
        
    except Exception as e:
        logger.error(f"Feature engineering pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()