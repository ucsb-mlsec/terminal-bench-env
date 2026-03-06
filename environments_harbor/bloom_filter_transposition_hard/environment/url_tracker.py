#!/usr/bin/env python3

import hashlib
import math


class URLTracker:
    """Memory-efficient URL tracker using Bloom filter"""
    
    def __init__(self, expected_items=100000, false_positive_rate=0.01):
        # Calculate optimal bloom filter parameters
        self.expected_items = expected_items
        self.false_positive_rate = false_positive_rate
        
        # Optimal bit array size: m = -(n * ln(p)) / (ln(2)^2)
        self.bit_array_size = int(-(expected_items * math.log(false_positive_rate)) / (math.log(2) ** 2))
        
        # Optimal number of hash functions: k = (m/n) * ln(2)
        self.num_hashes = int((self.bit_array_size / expected_items) * math.log(2))
        
        # Use bytearray for memory efficiency (1 bit per element)
        self.bit_array = bytearray((self.bit_array_size + 7) // 8)
    
    def _get_hash_values(self, url):
        """Generate multiple hash values for a URL"""
        # Use two hash functions and combine them to create multiple hashes
        hash1 = int(hashlib.md5(url.encode('utf-8')).hexdigest(), 16)
        hash2 = int(hashlib.sha1(url.encode('utf-8')).hexdigest(), 16)
        
        for i in range(self.num_hashes):
            # Double hashing technique: h(i) = h1 + i*h2
            yield (hash1 + i * hash2) % self.bit_array_size
    
    def add(self, url):
        """Add a URL to the bloom filter"""
        for hash_val in self._get_hash_values(url):
            byte_index = hash_val // 8
            bit_index = hash_val % 8
            self.bit_array[byte_index] |= (1 << bit_index)
    
    def contains(self, url):
        """Check if a URL might have been added (with possible false positives)"""
        for hash_val in self._get_hash_values(url):
            byte_index = hash_val // 8
            bit_index = hash_val % 8
            if not (self.bit_array[byte_index] & (1 << bit_index)):
                return False
        return True