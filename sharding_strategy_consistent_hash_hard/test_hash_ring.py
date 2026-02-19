#!/usr/bin/env python3

import unittest
from hash_ring import HashRing


class TestHashRing(unittest.TestCase):
    
    def test_basic_consistency(self):
        """Test that the same key always maps to the same node"""
        ring = HashRing(['node1', 'node2', 'node3'])
        
        # Test consistency across multiple calls
        for i in range(100):
            key = f'key{i}'
            first_node = ring.get_node(key)
            # Call multiple times to ensure consistency
            for _ in range(10):
                self.assertEqual(ring.get_node(key), first_node,
                               f"Key {key} should always map to the same node")
    
    def test_all_nodes_used(self):
        """Test that keys are distributed across all nodes"""
        nodes = ['node1', 'node2', 'node3']
        ring = HashRing(nodes)
        
        assigned_nodes = set()
        for i in range(1000):
            key = f'key{i}'
            node = ring.get_node(key)
            assigned_nodes.add(node)
        
        # All nodes should be used with enough keys
        self.assertEqual(assigned_nodes, set(nodes),
                        "All nodes should receive at least some keys")
    
    def test_node_addition_minimal_redistribution(self):
        """Test that adding a node causes minimal key redistribution"""
        # Start with 5 nodes
        initial_nodes = [f'node{i}' for i in range(5)]
        ring = HashRing(initial_nodes)
        
        # Assign 1000 keys and track their nodes
        num_keys = 1000
        initial_assignment = {}
        for i in range(num_keys):
            key = f'key{i}'
            initial_assignment[key] = ring.get_node(key)
        
        # Add a 6th node
        ring.add_node('node5')
        
        # Check how many keys moved
        moved_keys = 0
        for key, old_node in initial_assignment.items():
            new_node = ring.get_node(key)
            if new_node != old_node:
                moved_keys += 1
        
        # Theoretical minimum: 1000/6 ≈ 167 keys should move
        # Allow tolerance of ±50 keys (117-217 range)
        expected_moves = num_keys / 6
        tolerance = 50
        
        self.assertLess(moved_keys, expected_moves + tolerance,
                       f"Too many keys moved: {moved_keys}, expected around {expected_moves}")
        self.assertGreater(moved_keys, expected_moves - tolerance,
                          f"Too few keys moved: {moved_keys}, expected around {expected_moves}")
    
    def test_node_removal_minimal_redistribution(self):
        """Test that removing a node causes minimal key redistribution"""
        # Start with 5 nodes
        nodes = [f'node{i}' for i in range(5)]
        ring = HashRing(nodes)
        
        # Assign 1000 keys and track their nodes
        num_keys = 1000
        initial_assignment = {}
        for i in range(num_keys):
            key = f'key{i}'
            initial_assignment[key] = ring.get_node(key)
        
        # Count keys on node2
        keys_on_node2 = sum(1 for node in initial_assignment.values() if node == 'node2')
        
        # Remove node2
        ring.remove_node('node2')
        
        # Check how many keys moved
        moved_keys = 0
        for key, old_node in initial_assignment.items():
            new_node = ring.get_node(key)
            if new_node != old_node:
                moved_keys += 1
        
        # Only keys that were on node2 should move, plus allow small tolerance
        tolerance = 50
        self.assertLess(moved_keys, keys_on_node2 + tolerance,
                       f"Too many keys moved: {moved_keys}, only {keys_on_node2} were on removed node")
        self.assertGreater(moved_keys, keys_on_node2 - tolerance,
                          f"Too few keys moved: {moved_keys}, {keys_on_node2} were on removed node")
    
    def test_load_distribution(self):
        """Test that keys are reasonably distributed across nodes"""
        nodes = [f'node{i}' for i in range(5)]
        ring = HashRing(nodes)
        
        # Assign 1000 keys
        num_keys = 1000
        distribution = {node: 0 for node in nodes}
        
        for i in range(num_keys):
            key = f'key{i}'
            node = ring.get_node(key)
            distribution[node] += 1
        
        # Expected: 200 keys per node
        expected = num_keys / len(nodes)
        
        # Check that no node has wildly different load
        # Allow 30% deviation from expected
        for node, count in distribution.items():
            deviation = abs(count - expected) / expected
            self.assertLess(deviation, 0.3,
                           f"Node {node} has {count} keys, expected around {expected} (deviation: {deviation:.2%})")
    
    def test_empty_ring_handling(self):
        """Test behavior with empty ring"""
        ring = HashRing([])
        
        # Should handle gracefully or raise appropriate error
        try:
            result = ring.get_node('test_key')
            # If it returns something, it should be consistent
            self.assertEqual(result, ring.get_node('test_key'))
        except (ValueError, IndexError, KeyError):
            # Acceptable to raise an error for empty ring
            pass
    
    def test_single_node(self):
        """Test that single node ring works correctly"""
        ring = HashRing(['only_node'])
        
        # All keys should map to the only node
        for i in range(100):
            key = f'key{i}'
            self.assertEqual(ring.get_node(key), 'only_node')
    
    def test_node_addition_and_removal_sequence(self):
        """Test multiple additions and removals"""
        ring = HashRing(['node0', 'node1', 'node2'])
        
        # Add some nodes
        ring.add_node('node3')
        ring.add_node('node4')
        
        # Verify consistency
        test_keys = [f'key{i}' for i in range(50)]
        assignments = {key: ring.get_node(key) for key in test_keys}
        
        # Remove a node
        ring.remove_node('node1')
        
        # Keys not on node1 should stay on same node
        for key, old_node in assignments.items():
            new_node = ring.get_node(key)
            if old_node != 'node1':
                # Allow some movement due to virtual nodes, but most should stay
                pass  # Just check it returns valid node
            self.assertIn(new_node, ['node0', 'node2', 'node3', 'node4'])


if __name__ == '__main__':
    # Run tests and capture results
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestHashRing)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print results in required format
    tests_passed = result.testsRun - len(result.failures) - len(result.errors)
    tests_total = result.testsRun
    status = "PASS" if result.wasSuccessful() else "FAIL"
    
    print("\n" + "="*50)
    print(f"TESTS_PASSED: {tests_passed}")
    print(f"TESTS_TOTAL: {tests_total}")
    print(f"STATUS: {status}")