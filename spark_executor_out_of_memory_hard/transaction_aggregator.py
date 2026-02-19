#!/usr/bin/env python3

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def main():
    # Create Spark session for transaction aggregation
    spark = SparkSession.builder \
        .appName("TransactionAggregator") \
        .getOrCreate()
    
    print("Starting Transaction Aggregation Job...")
    
    # Read main transaction data - large dataset (500GB+)
    transactions_df = spark.read.parquet("/data/transactions/")
    print(f"Loaded transactions: {transactions_df.count()} records")
    
    # Read customer dimension table - also large
    customers_df = spark.read.parquet("/data/transactions/customers/")
    
    # Read product catalog - large product dataset
    products_df = spark.read.parquet("/data/transactions/products/")
    
    # Read regional store information
    stores_df = spark.read.parquet("/data/transactions/stores/")
    
    # MEMORY ANTI-PATTERN 1: Caching large DataFrames unnecessarily
    transactions_df.cache()
    customers_df.cache()
    
    # MEMORY ANTI-PATTERN 2: Multiple joins without filtering first
    # Join transactions with customers (full outer join on large tables)
    enriched_df = transactions_df.join(
        customers_df,
        transactions_df.customer_id == customers_df.customer_id,
        "left"
    )
    
    # MEMORY ANTI-PATTERN 3: Another large join without reducing data first
    enriched_df = enriched_df.join(
        products_df,
        enriched_df.product_id == products_df.product_id,
        "left"
    )
    
    # MEMORY ANTI-PATTERN 4: Third join creating even larger dataset
    enriched_df = enriched_df.join(
        stores_df,
        enriched_df.store_id == stores_df.store_id,
        "left"
    )
    
    # MEMORY ANTI-PATTERN 5: Broadcast join on large table instead of smaller one
    broadcast_df = F.broadcast(enriched_df)
    
    # Perform complex aggregations across multiple dimensions
    # This causes massive shuffle operations
    aggregated_by_customer = enriched_df.groupBy(
        "customer_id",
        "product_category",
        "region",
        F.to_date("transaction_date").alias("date")
    ).agg(
        F.sum("transaction_amount").alias("total_amount"),
        F.count("transaction_id").alias("transaction_count"),
        F.avg("transaction_amount").alias("avg_amount"),
        F.max("transaction_amount").alias("max_amount"),
        F.min("transaction_amount").alias("min_amount"),
        F.stddev("transaction_amount").alias("stddev_amount")
    )
    
    # MEMORY ANTI-PATTERN 6: Window function on large unsorted dataset
    # Creates additional shuffle and memory pressure
    window_spec = Window.partitionBy("customer_id", "region").orderBy("date")
    
    windowed_df = aggregated_by_customer.withColumn(
        "running_total",
        F.sum("total_amount").over(window_spec)
    ).withColumn(
        "rank",
        F.rank().over(window_spec)
    ).withColumn(
        "row_number",
        F.row_number().over(window_spec)
    )
    
    # MEMORY ANTI-PATTERN 7: Another groupBy creating more shuffle
    final_aggregation = windowed_df.groupBy(
        "customer_id",
        "region",
        "product_category"
    ).agg(
        F.sum("total_amount").alias("final_total"),
        F.max("running_total").alias("max_running_total"),
        F.count("*").alias("record_count"),
        F.collect_list("date").alias("transaction_dates"),  # Memory intensive
        F.collect_set("product_category").alias("categories")  # Memory intensive
    )
    
    # MEMORY ANTI-PATTERN 8: OrderBy on large dataset causing full sort shuffle
    sorted_results = final_aggregation.orderBy(
        F.desc("final_total"),
        "customer_id"
    )
    
    # MEMORY ANTI-PATTERN 9: Another complex calculation requiring shuffle
    percentile_df = sorted_results.withColumn(
        "percentile_rank",
        F.percent_rank().over(Window.orderBy(F.desc("final_total")))
    )
    
    # Create another large aggregation by region
    regional_summary = percentile_df.groupBy("region").agg(
        F.sum("final_total").alias("regional_total"),
        F.count("customer_id").alias("customer_count"),
        F.avg("final_total").alias("avg_customer_value")
    )
    
    # MEMORY ANTI-PATTERN 10: Unnecessary collect causing driver OOM risk
    # regional_stats = regional_summary.collect()  # Would cause driver OOM
    
    print("Writing aggregated results...")
    
    # Write results - using default partitioning which may be inefficient
    percentile_df.write \
        .mode("overwrite") \
        .parquet("/data/output/aggregated_transactions")
    
    regional_summary.write \
        .mode("overwrite") \
        .parquet("/data/output/regional_summary")
    
    print("Transaction aggregation completed successfully")
    
    # Unpersist cached dataframes
    transactions_df.unpersist()
    customers_df.unpersist()
    
    spark.stop()

if __name__ == "__main__":
    main()