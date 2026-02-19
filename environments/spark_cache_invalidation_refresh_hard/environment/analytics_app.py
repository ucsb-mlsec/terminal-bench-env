#!/usr/bin/env python3

from pyspark.sql import SparkSession
from pyspark import StorageLevel

def main():
    # Create SparkSession
    spark = SparkSession.builder \
        .appName('AnalyticsApp') \
        .master('local[*]') \
        .getOrCreate()
    
    # Load transactions data
    transactions_df = spark.read.csv(
        '/workspace/data/transactions.csv',
        header=True,
        inferSchema=True
    )
    
    # BUG 1: Cache without any refresh mechanism
    # When source file changes, this cached data will remain stale
    transactions_df.cache()
    
    # Load customers data
    customers_df = spark.read.csv(
        '/workspace/data/customers.csv',
        header=True,
        inferSchema=True
    )
    
    # BUG 3: Using inappropriate StorageLevel (MEMORY_AND_DISK_2 replicates unnecessarily)
    # This causes memory issues in single-node setup and is inefficient
    customers_df.persist(StorageLevel.MEMORY_AND_DISK_2)
    
    # BUG 2: Create derived cached DataFrame without invalidation logic
    # When parent DataFrames change, this joined cache becomes stale
    joined_df = transactions_df.join(
        customers_df,
        transactions_df.customer_id == customers_df.customer_id,
        'inner'
    )
    joined_df.cache()
    
    # Perform analytics queries
    print("=== Analytics Results ===")
    
    # Total transactions count
    total_transactions = transactions_df.count()
    print(f"Total Transactions: {total_transactions}")
    
    # Sum of transaction amounts
    total_amount = transactions_df.selectExpr("sum(amount) as total").collect()[0]['total']
    print(f"Total Amount: {total_amount}")
    
    # Top customers by spending
    print("\nTop Customers by Spending:")
    top_customers = joined_df.groupBy('customers.customer_id', 'name') \
        .agg({'amount': 'sum'}) \
        .withColumnRenamed('sum(amount)', 'total_spending') \
        .orderBy('total_spending', ascending=False) \
        .limit(5)
    
    top_customers.show()
    
    # Average transaction amount
    avg_amount = transactions_df.selectExpr("avg(amount) as average").collect()[0]['average']
    print(f"\nAverage Transaction Amount: {avg_amount}")
    
    # Customer count
    customer_count = customers_df.count()
    print(f"Total Customers: {customer_count}")
    
    # Note: Missing unpersist() calls means caches persist even when no longer needed
    # Note: No mechanism to detect source file changes and refresh caches
    
    spark.stop()

if __name__ == '__main__':
    main()