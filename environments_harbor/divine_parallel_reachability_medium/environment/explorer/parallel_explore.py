#!/usr/bin/env python3

import json
import threading
import os
from collections import deque

# Global shared data structures (accessed without proper locking - BUG!)
visited = set()
to_explore = deque()
depths = {}
graph = {}

def load_graph(filepath):
    """Load the state graph from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Build adjacency list
    adj = {}
    for state in data['states']:
        adj[state] = []
    
    for transition in data['transitions']:
        src, dst = transition
        if src in adj:
            adj[src].append(dst)
        else:
            adj[src] = [dst]
    
    return adj

def worker_explore(worker_id):
    """Worker function that explores states from the queue."""
    global visited, to_explore, depths, graph
    
    while True:
        # BUG: No locking when checking queue
        if len(to_explore) == 0:
            break
        
        # BUG: Race condition - multiple threads may pop the same item
        try:
            current_state = to_explore.popleft()
        except IndexError:
            break
        
        # BUG: Check and add to visited without atomic operation
        if current_state in visited:
            continue
        
        # BUG: Race condition - multiple threads may add same state
        visited.add(current_state)
        
        current_depth = depths.get(current_state, 0)
        
        # Explore neighbors
        if current_state in graph:
            for neighbor in graph[current_state]:
                # BUG: Race condition on visited check
                if neighbor not in visited:
                    # BUG: Multiple threads may add the same neighbor
                    to_explore.append(neighbor)
                    
                    # BUG: Race condition on depths dictionary
                    if neighbor not in depths:
                        depths[neighbor] = current_depth + 1

def parallel_explore(graph_data, num_workers=4):
    """Perform parallel state space exploration."""
    global visited, to_explore, depths, graph
    
    graph = graph_data
    visited = set()
    to_explore = deque()
    depths = {}
    
    # Start from initial state 0
    to_explore.append(0)
    depths[0] = 0
    
    # Create and start worker threads
    threads = []
    for i in range(num_workers):
        t = threading.Thread(target=worker_explore, args=(i,))
        threads.append(t)
        t.start()
    
    # Wait for all workers to complete
    for t in threads:
        t.join()
    
    # Calculate results
    reachable_count = len(visited)
    max_depth = max(depths.values()) if depths else 0
    
    return reachable_count, num_workers, max_depth

def main():
    # Load the state graph
    graph_path = '/workspace/data/state_graph.json'
    graph_data = load_graph(graph_path)
    
    # Perform parallel exploration with 4 workers
    reachable_count, worker_count, max_depth = parallel_explore(graph_data, num_workers=4)
    
    # Ensure output directory exists
    os.makedirs('/workspace/solution', exist_ok=True)
    
    # Write results
    output_path = '/workspace/solution/output.txt'
    with open(output_path, 'w') as f:
        f.write(f"{reachable_count} {worker_count} {max_depth}\n")

if __name__ == '__main__':
    main()