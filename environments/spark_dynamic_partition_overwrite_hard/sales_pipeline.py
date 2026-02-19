#!/usr/bin/env python3

from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder \
    .appName('SalesPipeline') \
    .master('local[*]') \
    .getOrCreate()

# THE BUG: Missing or incorrect partitionOverwriteMode configuration
# This causes the entire date partition to be overwritten instead of just the specific store partition
# Should be set to 'dynamic' but is either missing or set to 'static'
spark.conf.set('spark.sql.sources.partitionOverwriteMode', 'static')

# Read sales data from CSV
sales_df = spark.read \
    .option('header', True) \
    .option('inferSchema', True) \
    .csv('/workspace/input/sales_update.csv')

# Write to partitioned directory structure
# With static mode (or no mode configured), this overwrites the entire date=X partition
# when writing data for a single store
sales_df.write \
    .mode('overwrite') \
    .partitionBy('date', 'store') \
    .parquet('/workspace/output/sales/')

# Stop Spark session
spark.stop()