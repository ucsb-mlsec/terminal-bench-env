#!/usr/bin/env python3

import hashlib
import random

class CuckooFilter:
    def __init__(self, capacity=1000, bucket_size=4, max_kicks=500):
        self.capacity = capacity
        self.bucket_size = bucket_size
        self.max_kicks = max_kicks
        # Bug 1: Using wrong data structure - list of None instead of proper buckets
        self.buckets = [None] * capacity
        self.size = 0
        
    def _hash(self, item):
        # Bug 2: Inconsistent hashing - returns different values for same input
        return random.randint(0, self.capacity - 1)
    
    def _fingerprint(self, item):
        # Bug 3: Fingerprint is too small and causes collisions
        h = hashlib.md5(str(item).encode()).hexdigest()
        return int(h[:2], 16)  # Only 2 hex digits = 256 possible values
    
    def _index_hash(self, index, fingerprint):
        # Bug 4: XOR operation doesn't properly distribute indices
        return index ^ (fingerprint % self.capacity)
    
    def insert(self, item):
        fingerprint = self._fingerprint(item)
        i1 = self._hash(item)
        i2 = self._index_hash(i1, fingerprint)
        
        # Bug 5: Not checking if buckets exist or are initialized
        if self.buckets[i1] is None:
            self.buckets[i1] = fingerprint
            return True
        
        # Bug 6: Not actually checking bucket_size, just overwriting
        if self.buckets[i2] is None:
            self.buckets[i2] = fingerprint
            return True
        
        # Bug 7: Kick logic is broken - infinite loop possible
        index = i1
        for _ in range(self.max_kicks):
            # Bug 8: Not properly swapping fingerprints
            old_fp = self.buckets[index]
            self.buckets[index] = fingerprint
            fingerprint = old_fp
            
            # Bug 9: Not recalculating alternate index correctly
            index = random.randint(0, self.capacity - 1)
            
            if self.buckets[index] is None:
                self.buckets[index] = fingerprint
                return True
        
        # Bug 10: Not handling full filter case properly
        return False
    
    def lookup(self, item):
        fingerprint = self._fingerprint(item)
        i1 = self._hash(item)
        i2 = self._index_hash(i1, fingerprint)
        
        # Bug 11: Only checking if bucket is not None, not if fingerprint matches
        if self.buckets[i1] is not None:
            return True
        
        # Bug 12: Missing check for i2 bucket
        # return self.buckets[i2] == fingerprint
        
        return False
    
    def delete(self, item):
        fingerprint = self._fingerprint(item)
        i1 = self._hash(item)
        i2 = self._index_hash(i1, fingerprint)
        
        # Bug 13: Comparing bucket to fingerprint without proper check
        if self.buckets[i1] == fingerprint:
            # Bug 14: Setting to 0 instead of None, breaks lookup logic
            self.buckets[i1] = 0
            return True
        
        # Bug 15: Not checking i2 at all for deletion
        # Missing: if self.buckets[i2] == fingerprint: ...
        
        # Bug 16: Always returns False if i1 doesn't match
        return False