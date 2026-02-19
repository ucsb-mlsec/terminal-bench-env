#!/usr/bin/env python3

import yaml
import sys
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import dask.array as da

# Read cluster configuration
# BUG: Wrong file path
with open('/workspace/cluster_cfg.yaml', 'r') as f:
    config = yaml.safe_load(f)

# BUG: Not extracting the right part of config
# Should be config['cluster'] or similar
print("Creating SLURM cluster...")
cluster = SLURMCluster(config)

# BUG: Missing scale() call to request workers
# Should be cluster.scale(jobs=4) or similar
print("Cluster created")

# BUG: Not creating client with cluster
# Should be Client(cluster)
client = Client()

# Submit a distributed computation
# BUG: Not using dask properly for distributed work
print("Running computation...")
x = da.arange(1000, chunks=100)
# BUG: Missing .compute() call
result = x.sum()

# BUG: Trying to print without computing
print(f"Result: {result}")

# Check number of workers
# BUG: This will fail because workers weren't properly requested
workers = len(client.scheduler_info()['workers'])
print(f"Number of workers: {workers}")

# BUG: Missing cleanup - no cluster.close() or client.close()
print("Computation complete")

# BUG: No error handling anywhere
# BUG: No validation of configuration
# BUG: No proper output to results file