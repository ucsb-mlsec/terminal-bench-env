#!/usr/bin/env python3

import dask
import dask.array as da
import dask.dataframe as dd
from dask.delayed import delayed
from dask.distributed import Client
import numpy as np
import pandas as pd

# Initialize Dask client
client = Client()

# Phase 1: Array processing
x = da.random.random((5000, 5000), chunks=(1000, 1000))
y = da.sin(x) + da.cos(x)
z = da.exp(da.sqrt(da.abs(y)))
result1 = z.sum().compute()
# Phase 1 complete

# Phase 2: DataFrame aggregation
df = dd.from_pandas(
    pd.DataFrame({
        'key': np.random.randint(0, 100, 1000000),
        'value1': np.random.randn(1000000),
        'value2': np.random.randn(1000000),
        'value3': np.random.randn(1000000),
        'value4': np.random.randn(1000000),
        'value5': np.random.randn(1000000),
        'value6': np.random.randn(1000000),
        'value7': np.random.randn(1000000),
        'value8': np.random.randn(1000000),
        'value9': np.random.randn(1000000),
    }),
    npartitions=10
)
result2 = df.groupby('key').agg({
    'value1': 'sum',
    'value2': 'mean',
    'value3': 'count',
    'value4': 'sum',
    'value5': 'mean',
}).compute()
# Phase 2 complete

# Phase 3: Parallel task execution
@delayed
def compute_task(i):
    arr = np.random.randn(10000)
    return np.sum(arr ** 2) + np.mean(arr) * i

tasks = [compute_task(i) for i in range(100)]
result3 = dask.compute(*tasks)
# Phase 3 complete

# Close Dask client
client.close()