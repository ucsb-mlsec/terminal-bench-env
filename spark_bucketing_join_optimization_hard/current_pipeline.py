#!/usr/bin/env python3

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum as spark_sum, avg, to_date
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType

# TODO: Performance issues - joins causing shuffle operations
# TODO: SLA violations during peak hours

def create_spark_session():
    """Create SparkSession without bucketing configurations"""
    spark = SparkSession.builder \
        .appName("CustomerTransactionPipeline") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()
    return spark


def read_customer_profiles(spark, data_path="/data/"):
    """Read customer profile data from parquet files"""
    df_customers = spark.read.parquet(f"{data_path}customer_profiles.parquet")
    return df_customers


def read_transactions(spark, data_path="/data/"):
    """Read transaction data from parquet files"""
    df_transactions = spark.read.parquet(f"{data_path}transactions.parquet")
    return df_transactions


def perform_customer_transaction_join(df_customers, df_transactions):
    """
    Join customers and transactions on customer_id
    This operation causes expensive shuffle operations across the cluster
    """
    # Regular DataFrame join - causes shuffle on both sides
    df_joined = df_transactions.join(
        df_customers,
        on="customer_id",
        how="inner"
    )
    return df_joined


def aggregate_transaction_metrics(df_joined):
    """
    Aggregate transactions by date and customer
    Additional shuffles occur during groupBy operations
    """
    df_aggregated = df_joined.groupBy("transaction_date", "customer_id") \
        .agg(
            count("*").alias("transaction_count"),
            spark_sum("transaction_amount").alias("total_amount"),
            avg("transaction_amount").alias("avg_amount")
        )
    return df_aggregated


def filter_high_value_customers(df_joined, min_transaction_amount=1000.0):
    """Filter transactions for high-value customers"""
    df_filtered = df_joined.filter(
        col("transaction_amount") >= min_transaction_amount
    )
    return df_filtered


def write_results_to_parquet(df_aggregated, df_filtered, output_path="/data/output/"):
    """Write results without bucketing - standard parquet write"""
    # Write aggregated results
    df_aggregated.write \
        .mode("overwrite") \
        .parquet(f"{output_path}aggregated_transactions.parquet")
    
    # Write filtered high-value transactions
    df_filtered.write \
        .mode("overwrite") \
        .parquet(f"{output_path}high_value_transactions.parquet")


def main():
    """Main pipeline execution"""
    # Initialize Spark session
    spark = create_spark_session()
    
    try:
        # Read source data
        print("Reading customer profiles...")
        df_customers = read_customer_profiles(spark)
        
        print("Reading transaction data...")
        df_transactions = read_transactions(spark)
        
        # TODO: Performance issues - joins causing shuffle operations
        # This join operation triggers expensive shuffles on both datasets
        print("Performing customer-transaction join...")
        df_joined = perform_customer_transaction_join(df_customers, df_transactions)
        
        # Cache joined data for multiple downstream operations
        df_joined.cache()
        
        # Aggregate metrics by date and customer
        print("Aggregating transaction metrics...")
        df_aggregated = aggregate_transaction_metrics(df_joined)
        
        # Filter for high-value customers
        print("Filtering high-value customer transactions...")
        df_filtered = filter_high_value_customers(df_joined, min_transaction_amount=1000.0)
        
        # Write results to parquet without bucketing
        print("Writing results to parquet files...")
        write_results_to_parquet(df_aggregated, df_filtered)
        
        # Show execution plan to demonstrate shuffle operations
        print("\n=== Execution Plan (showing shuffle operations) ===")
        df_joined.explain(extended=True)
        
        print("\nPipeline completed successfully")
        
    except Exception as e:
        print(f"Pipeline failed with error: {str(e)}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()