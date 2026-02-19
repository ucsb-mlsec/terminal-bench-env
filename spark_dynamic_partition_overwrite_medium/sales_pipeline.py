#!/usr/bin/env python3

from pyspark.sql import SparkSession

# Create SparkSession
spark = SparkSession.builder \
    .appName('SalesPipeline') \
    .getOrCreate()

# Read incoming sales data
incoming_data = spark.read.csv(
    '/data/incoming/daily_sales.csv',
    header=True,
    inferSchema=True
)

# Write to partitioned dataset
# THE BUG: Using mode='overwrite' without dynamic partition overwrite
# This wipes out ALL existing partitions instead of just the relevant ones
incoming_data.write \
    .mode('overwrite') \
    .partitionBy('region', 'date') \
    .csv('/data/sales/')

spark.stop()