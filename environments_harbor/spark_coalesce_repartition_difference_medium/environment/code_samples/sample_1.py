#!/usr/bin/env python3

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Create SparkSession
spark = SparkSession.builder \
    .appName("DataFilteringOperation") \
    .master("local[*]") \
    .getOrCreate()

# Create a DataFrame with simulated data
# Simulating a large dataset with 10000 rows
data = [(i, f"user_{i}", i * 10, i % 100) for i in range(1, 10001)]
df = spark.createDataFrame(data, ["id", "username", "value", "category"])

print(f"Original DataFrame count: {df.count()}")
print(f"Original partitions: {df.rdd.getNumPartitions()}")

# Perform filtering operation that significantly reduces data volume
# This filter will keep only rows where value > 8000, removing ~80% of data
filtered_df = df.filter(col('value') > 8000)

print(f"Filtered DataFrame count: {filtered_df.count()}")

# Current partitioning operation
# Using repartition() after filtering - this causes a full shuffle
# which is expensive when we've already reduced the data volume
result_df = filtered_df.repartition(10)

print(f"Final partitions: {result_df.rdd.getNumPartitions()}")

# Show sample results
result_df.show(10)

# Note: After filtering that removes 80% of data, repartition() causes
# an unnecessary full shuffle. Coalesce would be more efficient here
# as it only reduces partitions without a full shuffle.

spark.stop()