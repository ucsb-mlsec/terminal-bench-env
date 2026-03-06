#!/usr/bin/env python3

import numpy as np

class VectorQuantizer:
    def __init__(self, num_codes, vector_dim):
        """Initialize the vector quantizer with a random codebook.
        
        Args:
            num_codes: Number of codebook entries
            vector_dim: Dimensionality of each vector
        """
        self.num_codes = num_codes
        self.vector_dim = vector_dim
        # Initialize codebook with random values
        np.random.seed(42)
        self.codebook = np.random.randn(num_codes, vector_dim)
    
    def quantize(self, vectors):
        """Quantize input vectors to nearest codebook entries.
        
        Args:
            vectors: NumPy array of shape (n_vectors, vector_dim)
            
        Returns:
            quantized_vectors: Array of shape (n_vectors, vector_dim) with codebook entries
            indices: Array of shape (n_vectors,) with codebook indices
        """
        n_vectors = vectors.shape[0]
        
        # Bug 1: Incorrect distance calculation - using sum instead of proper L2 distance
        # This will not give correct nearest neighbor results
        distances = np.zeros((n_vectors, self.num_codes))
        for i in range(n_vectors):
            for j in range(self.num_codes):
                # Wrong: should be np.sum((vectors[i] - self.codebook[j])**2)
                # or use broadcasting for efficiency
                distances[i, j] = np.sum(vectors[i] - self.codebook[j])
        
        # Bug 2: Using argmax instead of argmin - finding furthest instead of nearest
        indices = np.argmax(distances, axis=1)
        
        # Bug 3: Adding 1 to indices (off-by-one error) which makes them out of range
        indices = indices + 1
        
        # Bug 4: Returning the original vectors instead of codebook entries
        # This means quantized vectors won't match the codebook
        quantized_vectors = vectors.copy()
        
        return quantized_vectors, indices