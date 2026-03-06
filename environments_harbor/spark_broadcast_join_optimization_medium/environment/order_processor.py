#!/usr/bin/env python3

from pyspark.sql import SparkSession
import time

def main():
    # Create SparkSession
    spark = SparkSession.builder \
        .appName("OrderProcessor") \
        .master("local[*]") \
        .getOrCreate()
    
    print("Starting order processing...")
    
    # Read the datasets
    orders = spark.read.parquet('/workspace/data/orders.parquet')
    products = spark.read.parquet('/workspace/data/products.parquet')
    
    print(f"Orders count: {orders.count()}")
    print(f"Products count: {products.count()}")
    
    # Perform standard join (inefficient - causes shuffle)
    # This treats both datasets equally without considering size difference
    start_time = time.time()
    
    joined_data = orders.join(products, 'product_id')
    
    # Trigger computation
    result_count = joined_data.count()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("Join completed")
    print(f"Total records: {result_count}")
    print(f"Execution time: {execution_time:.2f} seconds")
    
    # Write results
    joined_data.write.mode('overwrite').parquet('/workspace/output/current_results.parquet')
    
    print("Results written to /workspace/output/current_results.parquet")
    
    spark.stop()

if __name__ == "__main__":
    main()