#!/usr/bin/env python3

import dask.array as da
import numpy as np

# Load data from file
data = np.load('/workspace/data.npy')

# Convert to Dask array with chunks
arr = da.from_array(data, chunks=(20, 20))

# Operation 1: Square all elements
step1 = arr ** 2

# Operation 2: Add 10 to all elements
step2 = step1 + 10

# Operation 3: Take square root of all elements
step3 = da.sqrt(step2)

# Execute the computation
result = step3.compute()

print('Benchmark complete')
print(f'Result shape: {result.shape}')