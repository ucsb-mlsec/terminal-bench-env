#!/usr/bin/env python3

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, StringType
import time

# Create SparkSession in local mode
spark = SparkSession.builder \
    .appName("Slow Sales Pipeline") \
    .master("local[*]") \
    .getOrCreate()

# Define regular Python UDFs
def calculate_revenue(quantity, unit_price):
    """Calculate total revenue from quantity and unit price"""
    if quantity is None or unit_price is None:
        return None
    return float(quantity * unit_price)

def classify_price_tier(unit_price):
    """Classify price into tier categories"""
    if unit_price is None:
        return None
    if unit_price < 50:
        return 'low'
    elif unit_price <= 200:
        return 'medium'
    else:
        return 'high'

def get_seasonal_multiplier(sale_date):
    """Get seasonal multiplier based on month"""
    if sale_date is None:
        return None
    # Extract month from YYYY-MM-DD format
    month = int(sale_date.split('-')[1])
    if month in [11, 12]:
        return 1.2
    elif month in [6, 7, 8]:
        return 1.1
    else:
        return 1.0

# Register UDFs with appropriate return types
calculate_revenue_udf = udf(calculate_revenue, FloatType())
classify_price_tier_udf = udf(classify_price_tier, StringType())
get_seasonal_multiplier_udf = udf(get_seasonal_multiplier, FloatType())

# Start timing
start_time = time.time()

# Read CSV data
df = spark.read.csv('/data/sales.csv', header=True, inferSchema=True)

# Apply UDFs to create new columns
df_with_metrics = df \
    .withColumn('total_revenue', calculate_revenue_udf(df['quantity'], df['unit_price'])) \
    .withColumn('price_tier', classify_price_tier_udf(df['unit_price'])) \
    .withColumn('seasonal_multiplier', get_seasonal_multiplier_udf(df['sale_date']))

# Select required columns
result_df = df_with_metrics.select('sale_id', 'total_revenue', 'price_tier', 'seasonal_multiplier')

# Write results to CSV
result_df.write.mode('overwrite').csv('/output/results.csv', header=True)

# Calculate duration
end_time = time.time()
duration = end_time - start_time

print(f"Pipeline execution time: {duration:.2f} seconds")

# Stop SparkSession
spark.stop()