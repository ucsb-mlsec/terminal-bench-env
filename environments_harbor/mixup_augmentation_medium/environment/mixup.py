#!/usr/bin/env python3

import numpy as np
import json


def mixup_batch(images, labels, alpha=0.2):
    """
    Apply Mixup augmentation to a batch of images and labels.
    
    Args:
        images: numpy array of shape (N, H, W) or (N, H, W, C)
        labels: one-hot encoded labels of shape (N, num_classes)
        alpha: parameter for Beta distribution
    
    Returns:
        mixed_images: augmented images
        mixed_labels: augmented labels
    """
    batch_size = images.shape[0]
    
    # Sample mixing coefficients from Beta distribution
    if alpha > 0:
        lam = np.random.beta(alpha, alpha, batch_size)
    else:
        lam = np.ones(batch_size)
    
    # Reshape lambda for broadcasting
    if len(images.shape) == 3:  # (N, H, W)
        lam_img = lam[:, np.newaxis, np.newaxis]
    else:  # (N, H, W, C)
        lam_img = lam[:, np.newaxis, np.newaxis, np.newaxis]
    
    lam_label = lam[:, np.newaxis]
    
    # Shuffle indices for pairing
    indices = np.random.permutation(batch_size)
    
    # Mix images and labels
    mixed_images = lam_img * images + (1 - lam_img) * images[indices]
    mixed_labels = lam_label * labels + (1 - lam_label) * labels[indices]
    
    return mixed_images, mixed_labels


def verify_mixup():
    """Verify the correctness of mixup_batch implementation."""
    
    # Load test data
    data = np.load('/data/test_batch.npz')
    images = data['images']
    labels = data['labels']
    
    # Run mixup with alpha=0.2
    mixed_images, mixed_labels = mixup_batch(images, labels, alpha=0.2)
    
    # Initialize verification results
    results = {
        "shape_valid": False,
        "range_valid": False,
        "labels_valid": False,
        "mixing_confirmed": False
    }
    
    # 1. Shape preservation
    results["shape_valid"] = (mixed_images.shape == images.shape and 
                              mixed_labels.shape == labels.shape)
    
    # 2. Value range check
    results["range_valid"] = (np.all(mixed_images >= 0) and 
                              np.all(mixed_images <= 255))
    
    # 3. Label consistency (sum to 1.0)
    label_sums = np.sum(mixed_labels, axis=1)
    results["labels_valid"] = np.allclose(label_sums, 1.0, rtol=1e-5)
    
    # 4. Mixing verification - check if at least one mixed image differs from all inputs
    mixing_confirmed = False
    for mixed_img in mixed_images:
        is_different_from_all = True
        for orig_img in images:
            if np.allclose(mixed_img, orig_img, rtol=1e-5):
                is_different_from_all = False
                break
        if is_different_from_all:
            mixing_confirmed = True
            break
    
    results["mixing_confirmed"] = mixing_confirmed
    
    # Save results
    with open('/workspace/verification.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Verification complete. Results saved to /workspace/verification.json")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    verify_mixup()