#!/usr/bin/env python3
"""Generate deterministic image data for hyperparameter optimization task.
Uses fixed seed to ensure identical data across builds."""
import numpy as np
import os

os.makedirs('/workspace/data', exist_ok=True)

rng = np.random.RandomState(42)

# Training images: 10000 x 32 x 32 x 3, float32 in [0, 1]
train_images = rng.random((10000, 32, 32, 3)).astype(np.float32)
np.save('/workspace/data/train_images.npy', train_images)

# Training labels: 10000 ints in [0, 9]
train_labels = rng.randint(0, 10, 10000).astype(np.int64)
np.save('/workspace/data/train_labels.npy', train_labels)

# Validation images: 2000 x 32 x 32 x 3, float32 in [0, 1]
val_images = rng.random((2000, 32, 32, 3)).astype(np.float32)
np.save('/workspace/data/val_images.npy', val_images)

# Validation labels: 2000 ints in [0, 9]
val_labels = rng.randint(0, 10, 2000).astype(np.int64)
np.save('/workspace/data/val_labels.npy', val_labels)
