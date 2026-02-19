#!/usr/bin/env python3

import json
import sys

def load_json(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"VERIFICATION FAILED: File {filepath} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"VERIFICATION FAILED: Invalid JSON in {filepath}: {e}")
        sys.exit(1)

def verify_solution():
    # Load files
    game_tree = load_json('/workspace/game_tree.json')
    solution = load_json('/workspace/solution.json')
    
    # Extract data
    nodes = game_tree.get('nodes', {})
    winning_nodes = set(solution.get('winning_nodes', []))
    losing_nodes = set(solution.get('losing_nodes', []))
    
    # Check all nodes are present
    all_node_ids = set(nodes.keys())
    solution_nodes = winning_nodes | losing_nodes
    
    # Check for nodes in both lists
    overlap = winning_nodes & losing_nodes
    if overlap:
        print(f"VERIFICATION FAILED: Nodes appear in both winning and losing lists: {overlap}")
        sys.exit(1)
    
    # Check for missing nodes
    missing = all_node_ids - solution_nodes
    if missing:
        print(f"VERIFICATION FAILED: Nodes missing from solution: {missing}")
        sys.exit(1)
    
    # Check for extra nodes
    extra = solution_nodes - all_node_ids
    if extra:
        print(f"VERIFICATION FAILED: Solution contains non-existent nodes: {extra}")
        sys.exit(1)
    
    # Verify correctness with memoization
    memo = {}
    
    def evaluate_node(node_id, visiting):
        # Check if already computed
        if node_id in memo:
            return memo[node_id]
        
        # Cycle detection - if we're currently visiting this node, 
        # we need to use the solution's classification
        if node_id in visiting:
            # Use the solution's classification for cycles
            return node_id in winning_nodes
        
        node = nodes.get(node_id)
        if not node:
            print(f"VERIFICATION FAILED: Node {node_id} referenced but not found in tree")
            sys.exit(1)
        
        # Mark as visiting
        visiting.add(node_id)
        
        # Leaf node
        if 'value' in node:
            result = node['value'] == 1
        # OR node - winning if ANY child is winning
        elif node.get('type') == 'OR':
            children = node.get('children', [])
            if not children:
                result = False  # OR node with no children is losing
            else:
                result = any(evaluate_node(child, visiting) for child in children)
        # AND node - winning if ALL children are winning
        elif node.get('type') == 'AND':
            children = node.get('children', [])
            if not children:
                result = True  # AND node with no children is winning
            else:
                result = all(evaluate_node(child, visiting) for child in children)
        else:
            print(f"VERIFICATION FAILED: Node {node_id} has invalid type or missing value")
            sys.exit(1)
        
        # Remove from visiting
        visiting.remove(node_id)
        
        # Memoize
        memo[node_id] = result
        return result
    
    # Verify each node
    for node_id in all_node_ids:
        is_winning_actual = evaluate_node(node_id, set())
        is_winning_claimed = node_id in winning_nodes
        
        if is_winning_actual != is_winning_claimed:
            node = nodes[node_id]
            if 'value' in node:
                print(f"VERIFICATION FAILED: Leaf node {node_id} has incorrect classification (value={node['value']}, claimed={'winning' if is_winning_claimed else 'losing'})")
            else:
                print(f"VERIFICATION FAILED: Node {node_id} is marked as {'winning' if is_winning_claimed else 'losing'} but should be {'winning' if is_winning_actual else 'losing'}")
            sys.exit(1)
    
    print("VERIFICATION PASSED")
    sys.exit(0)

if __name__ == "__main__":
    verify_solution()