#!/usr/bin/env python3
"""Generate deterministic test data for batch inference optimization task.
Uses fixed seed to ensure identical data across builds."""
import numpy as np
import os

os.makedirs('/data/matrices', exist_ok=True)

# batch_data: seed 42, standard normal
rng_batch = np.random.RandomState(42)
batch_data = rng_batch.randn(50000, 512).astype(np.float64)
np.save('/data/matrices/batch_data.npy', batch_data)

# weights: seed 42, scaled by 0.1
rng_weights = np.random.RandomState(42)
weights = (rng_weights.randn(512, 512) * 0.1).astype(np.float64)
np.save('/data/matrices/weights.npy', weights)
