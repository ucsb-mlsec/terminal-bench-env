#!/usr/bin/env python3

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def query1(spark):
    """Query with isin() filter - matches the pattern"""
    data = [
        ('order_001', 'active', 150.00),
        ('order_002', 'pending', 200.00),
        ('order_003', 'completed', 300.00),
        ('order_004', 'cancelled', 100.00),
        ('order_005', 'active', 250.00)
    ]
    df = spark.createDataFrame(data, ['order_id', 'status', 'amount'])
    # Filter using isin() - this matches our optimization pattern
    filtered_df = df.filter(col('status').isin(['active', 'pending']))
    return filtered_df.select('order_id', 'status', 'amount')

def query2(spark):
    """Query with different filter pattern - does NOT match"""
    data = [
        ('txn_001', 'processed', 150.00),
        ('txn_002', 'processed', 50.00),
        ('txn_003', 'failed', 300.00),
        ('txn_004', 'processed', 75.00)
    ]
    df = spark.createDataFrame(data, ['transaction_id', 'status', 'amount'])
    # Filter using comparison operator - NOT the isin() pattern
    filtered_df = df.filter(col('amount') > 100)
    return filtered_df.select('transaction_id', 'status', 'amount')

def query3(spark):
    """Query with isin() filter - matches the pattern"""
    data = [
        ('user_001', 'verified', 'premium'),
        ('user_002', 'pending', 'basic'),
        ('user_003', 'blocked', 'premium'),
        ('user_004', 'verified', 'basic'),
        ('user_005', 'suspended', 'premium')
    ]
    df = spark.createDataFrame(data, ['user_id', 'status', 'tier'])
    # Filter using isin() - this matches our optimization pattern
    filtered_df = df.filter(col('status').isin(['verified', 'pending', 'suspended']))
    return filtered_df.select('user_id', 'status', 'tier')

if __name__ == "__main__":
    # Create SparkSession
    spark = SparkSession.builder \
        .appName("TestQueries") \
        .master("local[*]") \
        .getOrCreate()
    
    # Execute queries
    print("Executing Query 1...")
    result1 = query1(spark)
    result1.show()
    
    print("Executing Query 2...")
    result2 = query2(spark)
    result2.show()
    
    print("Executing Query 3...")
    result3 = query3(spark)
    result3.show()
    
    spark.stop()