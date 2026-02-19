#!/usr/bin/env python3

import time
import json
from typing import Optional, Tuple, Dict, List

class StorageNode:
    """Simulates a single storage node in the distributed system"""
    
    def __init__(self):
        self.store = {}
    
    def set_lock(self, lock_name: str, client_id: str, expiry_time: float) -> bool:
        """Set a lock on this node"""
        self.store[lock_name] = {
            'client_id': client_id,
            'expiry_time': expiry_time
        }
        return True
    
    def get_lock(self, lock_name: str) -> Optional[Tuple[str, float]]:
        """Get lock information from this node"""
        if lock_name not in self.store:
            return None
        lock_data = self.store[lock_name]
        return (lock_data['client_id'], lock_data['expiry_time'])
    
    def release_lock(self, lock_name: str, client_id: str) -> bool:
        """Release a lock from this node"""
        if lock_name in self.store:
            if self.store[lock_name]['client_id'] == client_id:
                del self.store[lock_name]
                return True
        return False
    
    def is_expired(self, lock_name: str) -> bool:
        """Check if a lock has expired"""
        if lock_name not in self.store:
            return True
        # BUG: Not properly checking expiration
        expiry_time = self.store[lock_name]['expiry_time']
        return time.time() > expiry_time


class DistributedLock:
    """Redlock-style distributed lock implementation with BUGS"""
    
    def __init__(self, lock_name: str, num_nodes: int = 5):
        self.lock_name = lock_name
        self.num_nodes = num_nodes
        self.nodes = [StorageNode() for _ in range(num_nodes)]
    
    def acquire(self, client_id: str, timeout: int = 10) -> bool:
        """
        Attempt to acquire the distributed lock
        BUGGY: Should require majority consensus but has critical bugs
        """
        # BUG 1: Expiry time calculation is wrong - using current time instead of future
        expiry_time = time.time()  # Should be time.time() + timeout
        
        successful_nodes = 0
        
        for node in self.nodes:
            # BUG 2: Not properly checking if lock is already held by another client
            lock_info = node.get_lock(self.lock_name)
            if lock_info:
                # Missing proper validation - just checking if expired
                if node.is_expired(self.lock_name):
                    node.set_lock(self.lock_name, client_id, expiry_time)
                    successful_nodes += 1
            else:
                node.set_lock(self.lock_name, client_id, expiry_time)
                successful_nodes += 1
        
        # BUG 3: Only requires 1 node instead of majority
        required_nodes = 1  # Should be (self.num_nodes // 2) + 1
        
        if successful_nodes >= required_nodes:
            return True
        else:
            # Failed to acquire, should clean up but doesn't properly
            self.release(client_id)
            return False
    
    def release(self, client_id: str) -> bool:
        """
        Release the distributed lock
        BUGGY: Doesn't clean up all nodes properly
        """
        released = 0
        
        # BUG 4: Only releases from first 2 nodes, not all nodes
        for node in self.nodes[:2]:  # Should be self.nodes (all nodes)
            if node.release_lock(self.lock_name, client_id):
                released += 1
        
        return released > 0
    
    def is_locked(self) -> bool:
        """Check if the lock is currently held by any client"""
        locked_count = 0
        
        for node in self.nodes:
            lock_info = node.get_lock(self.lock_name)
            if lock_info and not node.is_expired(self.lock_name):
                locked_count += 1
        
        # Consider locked if majority of nodes have the lock
        return locked_count >= (self.num_nodes // 2) + 1


def run_scenario(scenario: Dict) -> Dict:
    """
    Run a single test scenario
    
    Args:
        scenario: Dict containing 'clients', 'lock_name', 'timeout', 'expected_success'
    
    Returns:
        Dict with 'passed' (bool) and 'details' (str)
    """
    lock_name = scenario.get('lock_name', 'test_lock')
    timeout = scenario.get('timeout', 10)
    clients = scenario.get('clients', [])
    expected_success = scenario.get('expected_success', [])
    
    # Create a fresh distributed lock for this scenario
    dist_lock = DistributedLock(lock_name)
    
    results = {}
    
    # Simulate clients attempting to acquire the lock
    for client_id in clients:
        success = dist_lock.acquire(client_id, timeout)
        results[client_id] = success
    
    # Check if results match expectations
    actual_success = [client for client, success in results.items() if success]
    
    # For mutual exclusion, only one client should succeed
    if len(actual_success) <= 1 and set(actual_success) == set(expected_success):
        return {
            'passed': True,
            'details': f'Expected {expected_success}, got {actual_success}'
        }
    else:
        return {
            'passed': False,
            'details': f'Expected {expected_success}, got {actual_success}. Mutual exclusion violated!'
        }


if __name__ == '__main__':
    # Simple test
    print("Running basic lock test...")
    lock = DistributedLock("test_resource")
    
    client1_acquired = lock.acquire("client1", timeout=5)
    print(f"Client 1 acquire: {client1_acquired}")
    
    client2_acquired = lock.acquire("client2", timeout=5)
    print(f"Client 2 acquire (should fail): {client2_acquired}")
    
    if client1_acquired and client2_acquired:
        print("ERROR: Both clients acquired the lock! Mutual exclusion failed.")
    elif client1_acquired and not client2_acquired:
        print("SUCCESS: Only client1 acquired the lock.")
    else:
        print("UNEXPECTED: Neither client acquired the lock.")
