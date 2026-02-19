#!/usr/bin/env python3

import hashlib

class HashRing:
    def __init__(self, nodes, virtual_nodes=150):
        """Initialize with a list of node identifiers (strings)"""
        self.virtual_nodes = virtual_nodes
        self.ring = {}
        self.sorted_keys = []
        self.nodes = set()
        
        for node in nodes:
            self.add_node(node)
    
    def _hash(self, key):
        """Generate hash value for a key"""
        return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16)
    
    def get_node(self, key):
        """Return which node should handle the given key (string)"""
        if not self.ring:
            return None
        
        hash_key = self._hash(key)
        
        # Find the first node in the ring with hash >= hash_key
        for ring_key in self.sorted_keys:
            if ring_key >= hash_key:
                return self.ring[ring_key]
        
        # Wrap around to the first node
        return self.ring[self.sorted_keys[0]]
    
    def add_node(self, node):
        """Add a new node to the ring"""
        if node in self.nodes:
            return
        
        self.nodes.add(node)
        
        # Add virtual nodes for this physical node
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}:{i}"
            hash_key = self._hash(virtual_key)
            self.ring[hash_key] = node
        
        # Re-sort the keys
        self.sorted_keys = sorted(self.ring.keys())
    
    def remove_node(self, node):
        """Remove a node from the ring"""
        if node not in self.nodes:
            return
        
        self.nodes.remove(node)
        
        # Remove all virtual nodes for this physical node
        keys_to_remove = []
        for hash_key, node_name in self.ring.items():
            if node_name == node:
                keys_to_remove.append(hash_key)
        
        for hash_key in keys_to_remove:
            del self.ring[hash_key]
        
        # Re-sort the keys
        self.sorted_keys = sorted(self.ring.keys())