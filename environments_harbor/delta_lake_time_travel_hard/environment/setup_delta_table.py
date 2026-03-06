#!/usr/bin/env python3

import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from delta import configure_spark_with_delta_pip

# Create workspace directories
os.makedirs("/workspace/data", exist_ok=True)

# Clean up any existing data
if os.path.exists("/workspace/data/sales_records"):
    shutil.rmtree("/workspace/data/sales_records")

# Configure Spark with Delta Lake
builder = SparkSession.builder \
    .appName("DeltaLakeSetup") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.sql.warehouse.dir", "/workspace/data")

spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Define table path
table_path = "/workspace/data/sales_records"

print("Creating Delta Lake table with version history...")

# VERSION 0: Initial data load with all regions including WEST
print("\nVersion 0: Initial data load with all regions...")
initial_data = []

# Add WEST region records (45 records)
for i in range(1, 46):
    initial_data.append((f"T{i:04d}", f"P{(i % 10) + 1:03d}", (i % 20) + 1, 
                        float(100 + (i * 50) % 5000), "WEST"))

# Add EAST region records (30 records)
for i in range(46, 76):
    initial_data.append((f"T{i:04d}", f"P{(i % 10) + 1:03d}", (i % 20) + 1, 
                        float(100 + (i * 50) % 5000), "EAST"))

# Add NORTH region records (25 records)
for i in range(76, 101):
    initial_data.append((f"T{i:04d}", f"P{(i % 10) + 1:03d}", (i % 20) + 1, 
                        float(100 + (i * 50) % 5000), "NORTH"))

# Add SOUTH region records (30 records)
for i in range(101, 131):
    initial_data.append((f"T{i:04d}", f"P{(i % 10) + 1:03d}", (i % 20) + 1, 
                        float(100 + (i * 50) % 5000), "SOUTH"))

schema = StructType([
    StructField("transaction_id", StringType(), False),
    StructField("product_id", StringType(), False),
    StructField("quantity", IntegerType(), False),
    StructField("revenue", FloatType(), False),
    StructField("region", StringType(), False)
])

df_v0 = spark.createDataFrame(initial_data, schema)
df_v0.write.format("delta").mode("overwrite").save(table_path)

west_count_v0 = df_v0.filter(df_v0.region == "WEST").count()
print(f"Version 0 created: Total records = {df_v0.count()}, WEST records = {west_count_v0}")

# VERSION 1: Add more records including additional WEST records
print("\nVersion 1: Adding more records...")
additional_data = []

# Add more WEST region records (10 records)
for i in range(131, 141):
    additional_data.append((f"T{i:04d}", f"P{(i % 10) + 1:03d}", (i % 20) + 1, 
                           float(100 + (i * 50) % 5000), "WEST"))

# Add more EAST region records (15 records)
for i in range(141, 156):
    additional_data.append((f"T{i:04d}", f"P{(i % 10) + 1:03d}", (i % 20) + 1, 
                           float(100 + (i * 50) % 5000), "EAST"))

df_v1 = spark.createDataFrame(additional_data, schema)
df_v1.write.format("delta").mode("append").save(table_path)

df_check_v1 = spark.read.format("delta").load(table_path)
west_count_v1 = df_check_v1.filter(df_check_v1.region == "WEST").count()
print(f"Version 1 created: Total records = {df_check_v1.count()}, WEST records = {west_count_v1}")

# VERSION 2: Add even more records including WEST
print("\nVersion 2: Adding more records...")
more_data = []

# Add more WEST region records (8 records)
for i in range(156, 164):
    more_data.append((f"T{i:04d}", f"P{(i % 10) + 1:03d}", (i % 20) + 1, 
                     float(100 + (i * 50) % 5000), "WEST"))

# Add more NORTH region records (12 records)
for i in range(164, 176):
    more_data.append((f"T{i:04d}", f"P{(i % 10) + 1:03d}", (i % 20) + 1, 
                     float(100 + (i * 50) % 5000), "NORTH"))

df_v2 = spark.createDataFrame(more_data, schema)
df_v2.write.format("delta").mode("append").save(table_path)

df_check_v2 = spark.read.format("delta").load(table_path)
west_count_v2 = df_check_v2.filter(df_check_v2.region == "WEST").count()
print(f"Version 2 created: Total records = {df_check_v2.count()}, WEST records = {west_count_v2}")

# VERSION 3: Delete all WEST region records
print("\nVersion 3: Deleting all WEST region records...")
from delta.tables import DeltaTable

delta_table = DeltaTable.forPath(spark, table_path)
delta_table.delete("region = 'WEST'")

df_check_v3 = spark.read.format("delta").load(table_path)
west_count_v3 = df_check_v3.filter(df_check_v3.region == "WEST").count()
print(f"Version 3 created: Total records = {df_check_v3.count()}, WEST records = {west_count_v3}")

# Display version history
print("\n" + "="*60)
print("Delta Lake table version history:")
print("="*60)
history_df = delta_table.history()
history_df.select("version", "operation", "operationMetrics").show(truncate=False)

print("\nSetup complete! Delta Lake table created at:", table_path)
print(f"Total versions created: {history_df.count()}")

spark.stop()