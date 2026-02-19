#!/usr/bin/env python3

import json

class Node:
    def __init__(self, node_id, name):
        self.node_id = node_id
        self.name = name
        self.neighbors = []
    
    def add_neighbor(self, node):
        self.neighbors.append(node)

# Create 5 node instances
node_1 = Node('node_1', 'Router-A')
node_2 = Node('node_2', 'Router-B')
node_3 = Node('node_3', 'Router-C')
node_4 = Node('node_4', 'Router-D')
node_5 = Node('node_5', 'Router-E')

# Create bidirectional connections forming a cycle
# Node 1 <-> Node 2
node_1.add_neighbor(node_2)
node_2.add_neighbor(node_1)

# Node 2 <-> Node 3
node_2.add_neighbor(node_3)
node_3.add_neighbor(node_2)

# Node 3 <-> Node 4
node_3.add_neighbor(node_4)
node_4.add_neighbor(node_3)

# Node 4 <-> Node 5
node_4.add_neighbor(node_5)
node_5.add_neighbor(node_4)

# Node 5 <-> Node 1
node_5.add_neighbor(node_1)
node_1.add_neighbor(node_5)

# Store all nodes in a graph variable
graph = [node_1, node_2, node_3, node_4, node_5]

if __name__ == '__main__':
    # This will fail with recursion error due to circular references
    try:
        json_output = json.dumps(graph, default=lambda o: o.__dict__)
        print("Serialization successful!")
    except Exception as e:
        print(f"Serialization failed: {e}")