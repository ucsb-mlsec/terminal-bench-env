#!/usr/bin/env python3

import hashlib
import json
import bisect
from typing import List, Dict, Optional, Set

class ShardManager:
    """
    Cache shard manager implementing consistent hashing for minimal key redistribution.
    """
    
    def __init__(self, servers: List[str], virtual_nodes: int = 150):
        """
        Initialize the shard manager with a list of servers.
        
        Args:
            servers: List of server IDs
            virtual_nodes: Number of virtual nodes per physical server for better distribution
        """
        self.servers: Set[str] = set(servers) if servers else set()
        self.virtual_nodes = virtual_nodes
        self.hash_ring: List[int] = []
        self.ring_map: Dict[int, str] = {}
        
        # Build the initial hash ring
        self._build_hash_ring()
    
    def _hash_key(self, key: str) -> int:
        """
        Hash a key to an integer using MD5.
        
        Args:
            key: The key to hash
            
        Returns:
            Integer hash value
        """
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def _build_hash_ring(self):
        """
        Build the consistent hash ring with virtual nodes.
        """
        self.hash_ring = []
        self.ring_map = {}
        
        for server in self.servers:
            for i in range(self.virtual_nodes):
                # Create virtual node identifier
                virtual_key = f"{server}:vnode:{i}"
                hash_val = self._hash_key(virtual_key)
                
                self.hash_ring.append(hash_val)
                self.ring_map[hash_val] = server
        
        # Sort the hash ring for binary search
        self.hash_ring.sort()
    
    def add_server(self, server_id: str):
        """
        Add a new server to the cluster.
        
        Args:
            server_id: ID of the server to add
        """
        if not server_id:
            raise ValueError("Server ID cannot be empty")
        
        if server_id in self.servers:
            return  # Server already exists
        
        self.servers.add(server_id)
        
        # Add virtual nodes for the new server
        for i in range(self.virtual_nodes):
            virtual_key = f"{server_id}:vnode:{i}"
            hash_val = self._hash_key(virtual_key)
            
            # Insert into sorted ring using bisect
            pos = bisect.bisect_left(self.hash_ring, hash_val)
            self.hash_ring.insert(pos, hash_val)
            self.ring_map[hash_val] = server_id
    
    def remove_server(self, server_id: str):
        """
        Remove a server from the cluster.
        
        Args:
            server_id: ID of the server to remove
        """
        if not server_id:
            raise ValueError("Server ID cannot be empty")
        
        if server_id not in self.servers:
            return  # Server doesn't exist
        
        self.servers.discard(server_id)
        
        # Remove all virtual nodes for this server
        nodes_to_remove = []
        for hash_val, server in self.ring_map.items():
            if server == server_id:
                nodes_to_remove.append(hash_val)
        
        for hash_val in nodes_to_remove:
            del self.ring_map[hash_val]
            self.hash_ring.remove(hash_val)
    
    def get_server_for_key(self, key: str) -> Optional[str]:
        """
        Get the server responsible for a given key using consistent hashing.
        
        Args:
            key: The cache key
            
        Returns:
            Server ID or None if no servers available
        """
        if not self.servers or not self.hash_ring:
            return None
        
        if not key:
            raise ValueError("Key cannot be empty")
        
        # Hash the key
        key_hash = self._hash_key(key)
        
        # Find the first server on the ring with hash >= key_hash
        pos = bisect.bisect_right(self.hash_ring, key_hash)
        
        # Wrap around if necessary
        if pos == len(self.hash_ring):
            pos = 0
        
        ring_hash = self.hash_ring[pos]
        return self.ring_map[ring_hash]
    
    def get_all_servers(self) -> List[str]:
        """
        Get list of all active servers.
        
        Returns:
            List of server IDs
        """
        return sorted(list(self.servers))
    
    def get_key_distribution(self, keys: List[str]) -> Dict[str, int]:
        """
        Get the distribution of keys across servers.
        
        Args:
            keys: List of keys to check
            
        Returns:
            Dictionary mapping server IDs to key counts
        """
        distribution = {server: 0 for server in self.servers}
        
        for key in keys:
            server = self.get_server_for_key(key)
            if server:
                distribution[server] += 1
        
        return distribution


def run_validation_tests():
    """
    Run validation tests based on test scenarios and generate results.
    """
    import os
    
    # Load test scenarios
    try:
        with open('/workspace/test_scenarios.json', 'r') as f:
            test_scenarios = json.load(f)
    except FileNotFoundError:
        print("Test scenarios file not found, using default scenarios")
        test_scenarios = {
            "scenarios": [
                {
                    "name": "add_server",
                    "initial_servers": ["server1", "server2", "server3"],
                    "action": "add",
                    "server": "server4",
                    "test_keys": [f"key_{i}" for i in range(1000)]
                },
                {
                    "name": "remove_server",
                    "initial_servers": ["server1", "server2", "server3"],
                    "action": "remove",
                    "server": "server2",
                    "test_keys": [f"key_{i}" for i in range(1000)]
                },
                {
                    "name": "single_server",
                    "initial_servers": ["server1"],
                    "action": "none",
                    "test_keys": [f"key_{i}" for i in range(100)]
                },
                {
                    "name": "empty_cluster",
                    "initial_servers": [],
                    "action": "add",
                    "server": "server1",
                    "test_keys": [f"key_{i}" for i in range(100)]
                }
            ]
        }
    
    total_tests = 0
    passed_tests = 0
    redistribution_percentages = []
    
    for scenario in test_scenarios.get("scenarios", []):
        total_tests += 1
        
        try:
            initial_servers = scenario["initial_servers"]
            test_keys = scenario["test_keys"]
            
            # Create shard manager
            manager = ShardManager(initial_servers)
            
            # Get initial key distribution
            initial_mapping = {}
            for key in test_keys:
                server = manager.get_server_for_key(key)
                initial_mapping[key] = server
            
            # Perform action
            action = scenario.get("action", "none")
            if action == "add":
                manager.add_server(scenario["server"])
            elif action == "remove":
                manager.remove_server(scenario["server"])
            
            # Get new key distribution
            new_mapping = {}
            for key in test_keys:
                server = manager.get_server_for_key(key)
                new_mapping[key] = server
            
            # Calculate redistribution percentage
            redistributed = 0
            for key in test_keys:
                if initial_mapping[key] != new_mapping[key]:
                    redistributed += 1
            
            if len(test_keys) > 0:
                redistribution_pct = (redistributed / len(test_keys)) * 100
                redistribution_percentages.append(redistribution_pct)
            
            # Test consistency
            consistency_check = True
            for _ in range(3):
                for key in test_keys[:100]:
                    if manager.get_server_for_key(key) != new_mapping[key]:
                        consistency_check = False
                        break
            
            # Check if test passed
            if action == "none":
                # For no-action scenarios, just check consistency
                if consistency_check:
                    passed_tests += 1
            elif consistency_check and (len(test_keys) == 0 or redistribution_pct < 50):
                passed_tests += 1
            
        except Exception as e:
            print(f"Test {scenario.get('name', 'unknown')} failed with error: {e}")
    
    # Calculate average redistribution
    avg_redistribution = sum(redistribution_percentages) / len(redistribution_percentages) if redistribution_percentages else 0
    
    # Save results
    results = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "redistribution_percentage": round(avg_redistribution, 2)
    }
    
    with open('/workspace/validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Validation complete: {passed_tests}/{total_tests} tests passed")
    print(f"Average redistribution: {avg_redistribution:.2f}%")
    
    return results


if __name__ == "__main__":
    run_validation_tests()