#!/usr/bin/env python3

import dask
import dask.dataframe as dd
import pandas as pd
import json

# Intentionally inefficient implementation with poor task granularity
# This creates excessive computational overhead by:
# 1. Using way too many partitions (one per row)
# 2. Creating individual tasks for trivial operations
# 3. Applying operations row-by-row instead of vectorized

def process_single_value(value):
    """Trivial function wrapped in a task - creates unnecessary overhead"""
    return value * 1.0

def normalize_single_value(value):
    """Another unnecessary task boundary"""
    return value + 0.0

def transform_single_value(value):
    """Yet another task for a simple operation"""
    return value / 1.0

# Read the CSV with excessive partitioning - one partition per row!
# This creates thousands of tasks just for reading
df = dd.read_csv('/workspace/data/measurements.csv', blocksize='1KB')

# Apply transformations that create even more tasks
# Each map_partitions call on tiny partitions creates many tasks
df['sensor_a_processed'] = df['sensor_a'].map_partitions(
    lambda x: x.apply(process_single_value), 
    meta=('sensor_a', 'f8')
)

df['sensor_a_normalized'] = df['sensor_a_processed'].map_partitions(
    lambda x: x.apply(normalize_single_value),
    meta=('sensor_a_processed', 'f8')
)

df['sensor_a_transformed'] = df['sensor_a_normalized'].map_partitions(
    lambda x: x.apply(transform_single_value),
    meta=('sensor_a_normalized', 'f8')
)

df['sensor_b_processed'] = df['sensor_b'].map_partitions(
    lambda x: x.apply(process_single_value),
    meta=('sensor_b', 'f8')
)

df['sensor_b_normalized'] = df['sensor_b_processed'].map_partitions(
    lambda x: x.apply(normalize_single_value),
    meta=('sensor_b_processed', 'f8')
)

df['sensor_b_transformed'] = df['sensor_b_normalized'].map_partitions(
    lambda x: x.apply(transform_single_value),
    meta=('sensor_b_normalized', 'f8')
)

df['sensor_c_processed'] = df['sensor_c'].map_partitions(
    lambda x: x.apply(process_single_value),
    meta=('sensor_c', 'f8')
)

df['sensor_c_normalized'] = df['sensor_c_processed'].map_partitions(
    lambda x: x.apply(normalize_single_value),
    meta=('sensor_c_processed', 'f8')
)

df['sensor_c_transformed'] = df['sensor_c_normalized'].map_partitions(
    lambda x: x.apply(transform_single_value),
    meta=('sensor_c_normalized', 'f8')
)

# Calculate means - even this creates many tasks due to partition count
mean_a = df['sensor_a_transformed'].mean()
mean_b = df['sensor_b_transformed'].mean()
mean_c = df['sensor_c_transformed'].mean()

# Compute results
mean_a_result = mean_a.compute()
mean_b_result = mean_b.compute()
mean_c_result = mean_c.compute()

# Count tasks in the execution graph
# This should be in the thousands due to excessive partitioning
task_count = len(mean_a.__dask_graph__()) + len(mean_b.__dask_graph__()) + len(mean_c.__dask_graph__())

# Save results
output = {
    "mean_sensor_a": float(mean_a_result),
    "mean_sensor_b": float(mean_b_result),
    "mean_sensor_c": float(mean_c_result),
    "task_count": int(task_count)
}

with open('/workspace/original_output.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"Processing complete. Task count: {task_count}")
print(f"Mean sensor_a: {mean_a_result:.2f}")
print(f"Mean sensor_b: {mean_b_result:.2f}")
print(f"Mean sensor_c: {mean_c_result:.2f}")