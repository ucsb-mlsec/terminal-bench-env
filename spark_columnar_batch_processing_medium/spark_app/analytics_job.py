#!/usr/bin/env python3

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType

def main():
    # Create SparkSession
    spark = SparkSession.builder \
        .appName("SalesAnalytics") \
        .getOrCreate()
    
    # Read sales transaction data from multiple CSV files
    sales_df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv("/data/sales_*.csv")
    
    # Read product dimension data
    product_df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv("/data/products.csv")
    
    # Read customer dimension data
    customer_df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv("/data/customers.csv")
    
    # Filter sales data based on date range
    filtered_sales = sales_df.filter(
        F.col("transaction_date") >= "2023-01-01"
    ).filter(
        F.col("transaction_date") <= "2023-12-31"
    )
    
    # Perform aggregations on sales data by product
    product_aggregations = filtered_sales.groupBy("product_id", "region") \
        .agg(
            F.sum("sales_amount").alias("total_sales"),
            F.sum("quantity").alias("total_quantity"),
            F.count("transaction_id").alias("transaction_count"),
            F.avg("sales_amount").alias("avg_sales_amount"),
            F.min("transaction_date").alias("first_transaction_date"),
            F.max("transaction_date").alias("last_transaction_date")
        )
    
    # Perform aggregations by region
    regional_aggregations = filtered_sales.groupBy("region") \
        .agg(
            F.sum("sales_amount").alias("regional_total_sales"),
            F.countDistinct("customer_id").alias("unique_customers"),
            F.countDistinct("product_id").alias("unique_products")
        )
    
    # Join sales aggregations with product dimension
    enriched_product_sales = product_aggregations.join(
        product_df,
        on="product_id",
        how="left"
    )
    
    # Join with regional aggregations
    final_results = enriched_product_sales.join(
        regional_aggregations,
        on="region",
        how="left"
    )
    
    # Calculate additional metrics
    final_results = final_results.withColumn(
        "sales_percentage_of_region",
        (F.col("total_sales") / F.col("regional_total_sales")) * 100
    ).withColumn(
        "avg_quantity_per_transaction",
        F.col("total_quantity") / F.col("transaction_count")
    )
    
    # Filter for high-performing products
    high_performers = final_results.filter(
        F.col("total_sales") > 10000
    ).filter(
        F.col("transaction_count") > 50
    )
    
    # Order by total sales descending
    high_performers = high_performers.orderBy(
        F.col("total_sales").desc()
    )
    
    # Write results to Parquet format with partitioning
    high_performers.write \
        .mode("overwrite") \
        .partitionBy("region") \
        .parquet("/output/results.parquet")
    
    # Generate summary statistics
    summary_stats = filtered_sales.groupBy("region") \
        .agg(
            F.sum("sales_amount").alias("total_revenue"),
            F.avg("sales_amount").alias("avg_transaction_value"),
            F.count("*").alias("total_transactions")
        )
    
    # Write summary to separate location
    summary_stats.write \
        .mode("overwrite") \
        .parquet("/output/summary_stats.parquet")
    
    print("Analytics job completed successfully")
    spark.stop()

if __name__ == "__main__":
    main()
```

Now I'll create the Spark configuration file for columnar batch processing optimization:

```python
#!/usr/bin/env python3

import os

# Create the solution directory if it doesn't exist
os.makedirs("/workspace/solution", exist_ok=True)

# Define the Spark configuration for columnar batch processing optimization
config_content = """spark.sql.inMemoryColumnarStorage.batchSize=10000
spark.sql.parquet.enableVectorizedReader=true
spark.sql.orc.enableVectorizedReader=true
spark.sql.inMemoryColumnarStorage.compressed=true
spark.sql.parquet.columnarReaderBatchSize=4096
spark.sql.codegen.wholeStage=true"""

# Write the configuration file
with open("/workspace/solution/spark_config.conf", "w") as f:
    f.write(config_content)

print("Configuration file created successfully at /workspace/solution/spark_config.conf")