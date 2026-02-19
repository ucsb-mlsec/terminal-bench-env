#!/usr/bin/env python3
"""
Data Preprocessing Script for Customer Transaction Data
Processes raw transaction data and creates customer segments.
"""

import pandas as pd
import numpy as np
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
import sys


# Configure logging
def setup_logging(log_level='INFO'):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('preprocessing.log')
        ]
    )
    return logging.getLogger(__name__)


def load_schema(schema_path='config/schema.json'):
    """Load and return the data schema."""
    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        logging.info(f"Schema loaded from {schema_path}")
        return schema
    except FileNotFoundError:
        logging.warning(f"Schema file not found at {schema_path}, using default schema")
        return {
            'required_columns': [
                'transaction_id', 'customer_id', 'transaction_date',
                'amount', 'category', 'payment_method'
            ],
            'numeric_columns': ['amount'],
            'date_columns': ['transaction_date']
        }


def load_raw_data(data_path='data/raw_data.csv', schema=None):
    """
    Load raw transaction data and validate against schema.
    
    Args:
        data_path: Path to raw CSV file
        schema: Data schema dictionary for validation
        
    Returns:
        pandas DataFrame with validated data
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading raw data from {data_path}")
    
    try:
        # Load the CSV file
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} rows from raw data")
        
        # Validate schema if provided
        if schema:
            required_cols = schema.get('required_columns', [])
            missing_cols = set(required_cols) - set(df.columns)
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            logger.info("Schema validation passed")
            
            # Convert date columns
            for date_col in schema.get('date_columns', []):
                if date_col in df.columns:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    logger.info(f"Converted {date_col} to datetime")
            
            # Validate numeric columns
            for num_col in schema.get('numeric_columns', []):
                if num_col in df.columns:
                    df[num_col] = pd.to_numeric(df[num_col], errors='coerce')
                    logger.info(f"Converted {num_col} to numeric")
        
        # Remove rows with missing critical data
        initial_rows = len(df)
        df = df.dropna(subset=['customer_id', 'amount'])
        dropped_rows = initial_rows - len(df)
        
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows with missing critical data")
        
        logger.info(f"Data validation complete. Final row count: {len(df)}")
        return df
        
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def aggregate_transactions(df):
    """
    Aggregate transaction data by customer.
    
    Args:
        df: DataFrame with transaction-level data
        
    Returns:
        DataFrame with customer-level aggregated data
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting transaction aggregation")
    
    # Group by customer and calculate aggregations
    agg_dict = {
        'transaction_id': 'count',
        'amount': ['mean', 'sum'],
        'transaction_date': 'max'
    }
    
    # Handle optional columns
    if 'category' in df.columns:
        agg_dict['category'] = lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
    
    if 'payment_method' in df.columns:
        agg_dict['payment_method'] = lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
    
    aggregated = df.groupby('customer_id').agg(agg_dict).reset_index()
    
    # Flatten column names
    aggregated.columns = ['customer_id', 'total_transactions', 'avg_transaction_amount', 
                         'lifetime_value', 'last_purchase_date']
    
    if 'category' in df.columns:
        category_mode = df.groupby('customer_id')['category'].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
        ).reset_index()
        category_mode.columns = ['customer_id', 'preferred_category']
        aggregated = aggregated.merge(category_mode, on='customer_id', how='left')
    
    if 'payment_method' in df.columns:
        payment_mode = df.groupby('customer_id')['payment_method'].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
        ).reset_index()
        payment_mode.columns = ['customer_id', 'preferred_payment']
        aggregated = aggregated.merge(payment_mode, on='customer_id', how='left')
    
    logger.info(f"Aggregated {len(df)} transactions into {len(aggregated)} customer records")
    
    return aggregated


def calculate_segments(df):
    """
    Assign customer segments based on lifetime value.
    
    Args:
        df: DataFrame with customer aggregated data
        
    Returns:
        DataFrame with customer_segment column added
    """
    logger = logging.getLogger(__name__)
    logger.info("Calculating customer segments")
    
    def assign_segment(lifetime_value):
        if lifetime_value > 10000:
            return 'Premium'
        elif lifetime_value >= 1000:
            return 'Standard'
        elif lifetime_value >= 100:
            return 'Basic'
        else:
            return 'New'
    
    df['customer_segment'] = df['lifetime_value'].apply(assign_segment)
    
    # Log segment distribution
    segment_counts = df['customer_segment'].value_counts()
    logger.info("Customer segment distribution:")
    for segment, count in segment_counts.items():
        logger.info(f"  {segment}: {count} customers ({count/len(df)*100:.1f}%)")
    
    return df


def validate_output(df):
    """
    Validate the quality and schema compliance of processed data.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        bool: True if validation passes
    """
    logger = logging.getLogger(__name__)
    logger.info("Validating output data")
    
    validation_passed = True
    
    # Check for required columns
    required_cols = ['customer_id', 'total_transactions', 'avg_transaction_amount',
                    'lifetime_value', 'customer_segment']
    
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        validation_passed = False
    
    # Check for null values in critical columns
    for col in required_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                logger.warning(f"Column {col} has {null_count} null values")
    
    # Validate numeric ranges
    if 'lifetime_value' in df.columns:
        if (df['lifetime_value'] < 0).any():
            logger.error("Found negative lifetime_value")
            validation_passed = False
    
    if 'total_transactions' in df.columns:
        if (df['total_transactions'] < 1).any():
            logger.error("Found customers with less than 1 transaction")
            validation_passed = False
    
    # Check segment values
    if 'customer_segment' in df.columns:
        valid_segments = {'Premium', 'Standard', 'Basic', 'New'}
        invalid_segments = set(df['customer_segment'].unique()) - valid_segments
        if invalid_segments:
            logger.error(f"Found invalid segments: {invalid_segments}")
            validation_passed = False
    
    if validation_passed:
        logger.info("Output validation passed successfully")
    else:
        logger.error("Output validation failed")
    
    return validation_passed


def save_preprocessed(df, output_path='data/preprocessed_data.csv'):
    """
    Save preprocessed data to CSV file.
    
    Args:
        df: DataFrame to save
        output_path: Path where to save the file
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Saving preprocessed data to {output_path}")
    
    try:
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        file_size = Path(output_path).stat().st_size
        logger.info(f"Successfully saved {len(df)} records ({file_size:,} bytes)")
        
    except Exception as e:
        logger.error(f"Error saving preprocessed data: {str(e)}")
        raise


def main():
    """Main preprocessing pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Preprocess customer transaction data')
    parser.add_argument('--input', default='data/raw_data.csv', 
                       help='Path to raw data CSV file')
    parser.add_argument('--output', default='data/preprocessed_data.csv',
                       help='Path to output preprocessed CSV file')
    parser.add_argument('--schema', default='config/schema.json',
                       help='Path to schema JSON file')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("=" * 60)
    logger.info("Starting data preprocessing pipeline")
    logger.info("=" * 60)
    
    try:
        # Load schema
        schema = load_schema(args.schema)
        
        # Load raw data
        logger.info("Step 1: Loading raw data")
        raw_data = load_raw_data(args.input, schema)
        
        # Aggregate transactions
        logger.info("Step 2: Aggregating transactions by customer")
        aggregated_data = aggregate_transactions(raw_data)
        
        # Calculate segments
        logger.info("Step 3: Calculating customer segments")
        segmented_data = calculate_segments(aggregated_data)
        
        # Validate output
        logger.info("Step 4: Validating processed data")
        if not validate_output(segmented_data):
            logger.warning("Validation warnings present, but continuing with save")
        
        # Save preprocessed data
        logger.info("Step 5: Saving preprocessed data")
        save_preprocessed(segmented_data, args.output)
        
        logger.info("=" * 60)
        logger.info("Preprocessing pipeline completed successfully")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())