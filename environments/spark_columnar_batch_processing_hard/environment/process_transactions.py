#!/usr/bin/env python3

"""
Customer Transaction Processing Application
Processes batch transaction data using Apache Spark with columnar operations
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg
import sys


def create_spark_session():
    """
    Initialize Spark session with application-specific configuration
    """
    try:
        spark = SparkSession.builder \
            .appName("CustomerTransactionProcessor") \
            .getOrCreate()
        return spark
    except Exception as e:
        print(f"Error creating Spark session: {e}")
        sys.exit(1)


def load_data(spark, path):
    """
    Load transaction data from parquet files
    
    Args:
        spark: SparkSession object
        path: Path to parquet files
    
    Returns:
        DataFrame containing transaction data
    """
    try:
        df = spark.read.parquet(path)
        print(f"Successfully loaded data from {path}")
        return df
    except Exception as e:
        print(f"Error loading data from {path}: {e}")
        raise


def process_batch(df):
    """
    Process transaction batch with aggregations
    
    This function performs columnar batch operations:
    1. Filters high-value transactions (amount > 100)
    2. Groups data by customer_id for aggregation
    3. Calculates sum and average using vectorized operations
    
    Args:
        df: Input DataFrame with transaction data
    
    Returns:
        Aggregated DataFrame with customer-level metrics
    """
    try:
        # Filter transactions greater than 100 - leverages columnar filtering
        filtered_df = df.filter(col("amount") > 100)
        
        # Group by customer_id and perform batch aggregations
        # Vectorized operations process multiple rows simultaneously
        aggregated_df = filtered_df.groupBy("customer_id").agg(
            sum("amount").alias("total_amount"),
            avg("amount").alias("average_amount")
        )
        
        print(f"Processed batch: {aggregated_df.count()} customers")
        return aggregated_df
    except Exception as e:
        print(f"Error processing batch: {e}")
        raise


def save_results(df, output_path):
    """
    Save processed results to parquet format
    
    Args:
        df: DataFrame to save
        output_path: Destination path for output files
    """
    try:
        # Write using columnar parquet format for efficient storage
        df.write.mode("overwrite").parquet(output_path)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results to {output_path}: {e}")
        raise


def main():
    """
    Main execution flow for transaction processing pipeline
    """
    spark = None
    try:
        # Initialize Spark session
        spark = create_spark_session()
        
        # Load transaction data - reads columnar parquet format
        transactions_df = load_data(spark, "/data/transactions")
        
        # Process batch with aggregations using columnar execution
        processed_df = process_batch(transactions_df)
        
        # Save aggregated results
        save_results(processed_df, "/output/processed")
        
        print("Transaction processing completed successfully")
        
    except Exception as e:
        print(f"Application failed: {e}")
        sys.exit(1)
    finally:
        # Clean up Spark session
        if spark:
            spark.stop()
            print("Spark session stopped")


if __name__ == "__main__":
    main()