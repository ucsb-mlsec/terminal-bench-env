#!/usr/bin/env python3
from pyspark.sql import SparkSession

def load_orders():
    """
    Load order data from CSV files and cache it.
    BUG: The cached DataFrame is not invalidated when new files are added.
    """
    spark = SparkSession.builder \
        .appName("OrderAnalytics") \
        .master("local[*]") \
        .getOrCreate()
    
    # Read all CSV files from the orders directory
    orders_df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv("/data/orders/*.csv")
    
    # CACHING HAPPENS HERE - This is where the data gets cached
    orders_df.cache()
    
    return orders_df

def get_order_count(df):
    """
    Get the count of orders in the DataFrame.
    """
    return df.count()

if __name__ == "__main__":
    # BUG: Loading and caching happens only ONCE at the beginning
    # If new CSV files are added to /data/orders/ after this point,
    # they won't be included in the cached DataFrame
    orders_df = load_orders()
    
    # Print initial count
    initial_count = get_order_count(orders_df)
    print(f"Initial order count: {initial_count}")
    
    # The application continues to use this same cached DataFrame
    # even if new order files are added to the directory
    # This is the root cause of the data freshness issue