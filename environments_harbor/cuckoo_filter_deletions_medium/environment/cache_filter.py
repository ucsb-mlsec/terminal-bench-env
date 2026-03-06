#!/usr/bin/env python3

import hashlib
import random

class CuckooFilter:
    def __init__(self, capacity=1000, bucket_size=4, fingerprint_size=8):
        self.capacity = capacity
        self.bucket_size = bucket_size
        self.fingerprint_size = fingerprint_size
        self.num_buckets = capacity // bucket_size
        self.buckets = [[] for _ in range(self.num_buckets)]
        self.max_kicks = 500
        
    def _hash(self, item):
        """Hash function for items"""
        if isinstance(item, str):
            item = item.encode('utf-8')
        return int(hashlib.md5(item).hexdigest(), 16)
    
    def _fingerprint(self, item):
        """Generate fingerprint for an item"""
        hash_val = self._hash(item)
        # Use modulo to limit fingerprint size, ensure it's not zero
        fingerprint = (hash_val % (2 ** self.fingerprint_size - 1)) + 1
        return fingerprint
    
    def _index_hash(self, item):
        """Calculate primary bucket index"""
        hash_val = self._hash(item)
        return hash_val % self.num_buckets
    
    def _alt_index(self, index, fingerprint):
        """Calculate alternative bucket index"""
        # XOR-based alternative index calculation
        hash_val = index ^ self._hash(str(fingerprint).encode('utf-8'))
        return hash_val % self.num_buckets
    
    def insert(self, item):
        """Insert item into the filter"""
        fingerprint = self._fingerprint(item)
        i1 = self._index_hash(item)
        i2 = self._alt_index(i1, fingerprint)
        
        # Try to insert in the first bucket
        if len(self.buckets[i1]) < self.bucket_size:
            self.buckets[i1].append(fingerprint)
            return True
        
        # Try to insert in the alternative bucket
        if len(self.buckets[i2]) < self.bucket_size:
            self.buckets[i2].append(fingerprint)
            return True
        
        # Both buckets are full, start cuckoo process
        # Randomly pick one of the two buckets
        idx = random.choice([i1, i2])
        
        for _ in range(self.max_kicks):
            # Randomly evict an entry from the bucket
            evict_pos = random.randint(0, len(self.buckets[idx]) - 1)
            evicted_fingerprint = self.buckets[idx][evict_pos]
            self.buckets[idx][evict_pos] = fingerprint
            
            # Calculate alternative index for evicted fingerprint
            idx = self._alt_index(idx, evicted_fingerprint)
            fingerprint = evicted_fingerprint
            
            # Try to insert evicted fingerprint
            if len(self.buckets[idx]) < self.bucket_size:
                self.buckets[idx].append(fingerprint)
                return True
        
        # Failed to insert after max kicks
        return False
    
    def lookup(self, item):
        """Check if item might be in the filter"""
        fingerprint = self._fingerprint(item)
        i1 = self._index_hash(item)
        i2 = self._alt_index(i1, fingerprint)
        
        # Check both possible buckets
        if fingerprint in self.buckets[i1] or fingerprint in self.buckets[i2]:
            return True
        return False
    
    def delete(self, item):
        """
        BROKEN DELETE METHOD
        This implementation has several bugs:
        1. It deletes from BOTH buckets when it should only delete from one
        2. It removes ALL occurrences instead of just one
        3. It doesn't verify which bucket actually contains the item
        """
        fingerprint = self._fingerprint(item)
        i1 = self._index_hash(item)
        i2 = self._alt_index(i1, fingerprint)
        
        # BUG: This removes from BOTH buckets, corrupting the filter
        # It should only remove from the bucket where the item actually is
        deleted = False
        
        # BUG: Removes ALL occurrences from first bucket
        while fingerprint in self.buckets[i1]:
            self.buckets[i1].remove(fingerprint)
            deleted = True
        
        # BUG: Also removes ALL occurrences from second bucket
        while fingerprint in self.buckets[i2]:
            self.buckets[i2].remove(fingerprint)
            deleted = True
        
        return deleted