#!/usr/bin/env python3
"""Generate deterministic test matrix for cache-friendly data layout task.
Uses fixed seed to ensure identical data across builds."""
import numpy as np

rng = np.random.RandomState(12345)
matrix = rng.randn(8192, 8192).astype(np.float64)
matrix.tofile('/workspace/test_matrix.bin')
