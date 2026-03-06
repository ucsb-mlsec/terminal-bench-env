#!/usr/bin/env python3

from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel
from pyspark import SparkConf
import time

# Initialize Spark Session
def create_spark_session():
    """Create and configure Spark session for caching analysis"""
    conf = SparkConf()
    conf.set("spark.executor.memory", "4g")
    conf.set("spark.driver.memory", "2g")
    
    spark = SparkSession.builder \
        .appName("CachingAnalysis") \
        .config(conf=conf) \
        .getOrCreate()
    
    return spark

def process_datasets(spark):
    """Process three datasets with different caching requirements"""
    
    # ========== DATASET A ==========
    # Frequently accessed, fits in memory, but causing GC pressure
    # Current issue: MEMORY_ONLY causing excessive garbage collection
    # TODO: Fix caching strategy - need serialized format to reduce GC
    print("Loading Dataset A...")
    dataset_a = spark.read.csv("/data/dataset_a.csv", header=True, inferSchema=True)
    
    # Apply transformations
    dataset_a_transformed = dataset_a.filter(dataset_a["status"] == "active") \
        .select("id", "name", "value", "timestamp") \
        .withColumn("processed_value", dataset_a["value"] * 1.5)
    
    # Current storage level causing issues - MEMORY_ONLY causes GC pressure
    dataset_a_transformed.persist(StorageLevel.MEMORY_ONLY)
    
    # Frequent access operations
    print(f"Dataset A count: {dataset_a_transformed.count()}")
    dataset_a_transformed.show(10)
    
    
    # ========== DATASET B ==========
    # Large dataset exceeding available memory, causing evictions
    # Current issue: MEMORY_ONLY inappropriate for large datasets
    # TODO: Fix caching strategy - needs disk spillover capability
    print("Loading Dataset B...")
    dataset_b = spark.read.csv("/data/dataset_b.csv", header=True, inferSchema=True)
    
    # Heavy transformations on large dataset
    dataset_b_aggregated = dataset_b.groupBy("category", "region") \
        .agg({"amount": "sum", "quantity": "avg", "transactions": "count"}) \
        .filter("sum(amount) > 1000")
    
    # Current storage level causing issues - evictions due to size
    # MEMORY_ONLY is forcing evictions when dataset doesn't fit
    dataset_b_aggregated.persist(StorageLevel.MEMORY_ONLY)
    
    # Multiple operations requiring cached data
    print(f"Dataset B count: {dataset_b_aggregated.count()}")
    dataset_b_aggregated.show(20)
    top_categories = dataset_b_aggregated.orderBy("sum(amount)", ascending=False).take(5)
    print(f"Top 5 categories: {top_categories}")
    
    
    # ========== DATASET C ==========
    # Accessed across multiple executors, experiencing data locality issues
    # Current issue: No replication, causing data locality problems
    # TODO: Fix caching strategy - needs replication for better locality
    print("Loading Dataset C...")
    dataset_c = spark.read.csv("/data/dataset_c.csv", header=True, inferSchema=True)
    
    # Transformations requiring distributed access
    dataset_c_enriched = dataset_c.filter(dataset_c["priority"] == "high") \
        .select("user_id", "product_id", "score", "location")
    
    # Current storage level causing issues - no replication means poor locality
    # Single copy causes network shuffle when accessed from different executors
    dataset_c_enriched.persist(StorageLevel.MEMORY_ONLY)
    
    # Operations across multiple executors
    print(f"Dataset C count: {dataset_c_enriched.count()}")
    
    # Join operation that benefits from replication
    dataset_c_grouped = dataset_c_enriched.groupBy("location").count()
    dataset_c_grouped.show(15)
    
    # Cross-executor access pattern
    dataset_c_enriched.rdd.map(lambda x: (x.user_id, x.score)).reduceByKey(lambda a, b: a + b).collect()
    
    
    # ========== DEMONSTRATE PERFORMANCE ISSUES ==========
    print("\n=== Performance Issue Summary ===")
    print("Dataset A: GC pressure from storing deserialized objects")
    print("Dataset B: Cache evictions due to insufficient memory for large dataset")
    print("Dataset C: Poor data locality without replication across executors")
    
    # Simulate multiple access patterns
    time.sleep(2)
    dataset_a_transformed.count()
    dataset_b_aggregated.count()
    dataset_c_enriched.count()
    
    # Unpersist datasets
    dataset_a_transformed.unpersist()
    dataset_b_aggregated.unpersist()
    dataset_c_enriched.unpersist()

def main():
    """Main execution function"""
    spark = create_spark_session()
    
    try:
        process_datasets(spark)
    finally:
        spark.stop()

if __name__ == "__main__":
    main()