#!/usr/bin/env python3

import time
import json
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types
from pyspark.sql.functions import udf

def categorize_transaction(amount):
    """Categorize transaction based on amount ranges"""
    if amount < 100:
        return 'small'
    elif amount < 500:
        return 'medium'
    elif amount < 2000:
        return 'large'
    else:
        return 'very_large'

def run_original_approach():
    """Run the original approach using Python UDF"""
    try:
        spark = SparkSession.builder \
            .appName("TransactionProcessor") \
            .getOrCreate()
        
        # Register the UDF
        categorize_udf = udf(categorize_transaction, types.StringType())
        
        # Read the data
        df = spark.read.csv('/home/user/data/transactions.csv', 
                           header=True, 
                           inferSchema=True)
        
        # Measure execution time
        start_time = time.time()
        
        # Apply UDF and perform aggregation
        result_df = df.withColumn('category', categorize_udf(F.col('amount')))
        aggregated = result_df.groupBy('category').count()
        
        # Trigger execution
        aggregated.collect()
        
        end_time = time.time()
        execution_time_ms = int((end_time - start_time) * 1000)
        
        spark.stop()
        return execution_time_ms
        
    except Exception as e:
        print(f"Error in original approach: {e}")
        raise

def run_optimized_approach():
    """Run optimized approach using native Spark SQL expressions"""
    try:
        spark = SparkSession.builder \
            .appName("TransactionProcessor") \
            .getOrCreate()
        
        # Read the data
        df = spark.read.csv('/home/user/data/transactions.csv', 
                           header=True, 
                           inferSchema=True)
        
        # Measure execution time
        start_time = time.time()
        
        # Use native Spark SQL when expression instead of Python UDF
        result_df = df.withColumn('category',
            F.when(F.col('amount') < 100, 'small')
             .when(F.col('amount') < 500, 'medium')
             .when(F.col('amount') < 2000, 'large')
             .otherwise('very_large')
        )
        
        aggregated = result_df.groupBy('category').count()
        
        # Trigger execution
        aggregated.collect()
        
        end_time = time.time()
        execution_time_ms = int((end_time - start_time) * 1000)
        
        spark.stop()
        return execution_time_ms
        
    except Exception as e:
        print(f"Error in optimized approach: {e}")
        raise

def main():
    try:
        # Run original approach
        original_time = run_original_approach()
        print(f"Original execution time: {original_time} ms")
        
        # Run optimized approach
        optimized_time = run_optimized_approach()
        print(f"Optimized execution time: {optimized_time} ms")
        
        # Calculate improvement percentage
        improvement_percent = round(((original_time - optimized_time) / original_time) * 100, 1)
        
        # Prepare results
        results = {
            "original_execution_ms": original_time,
            "optimized_execution_ms": optimized_time,
            "performance_improvement_percent": improvement_percent
        }
        
        # Save to JSON file
        with open('/tmp/spark_performance.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Performance improvement: {improvement_percent}%")
        print("Results saved to /tmp/spark_performance.json")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()