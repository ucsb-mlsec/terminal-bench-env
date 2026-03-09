#!/usr/bin/env python3
"""Script to generate social_graph.txt data file."""
import random
import os

random.seed(42)

num_nodes = 50000
output_path = os.path.join(os.path.dirname(__file__), 'environment/data/social_graph.txt')

edges = []

# Hub nodes (indices 0-3749): high degree ~53 edges each
# 3750 * ~53 = ~200k edges total, so we'll use hubs for the bulk
# and small degrees for the rest to hit ~200k exactly

# Strategy:
# - 3750 hub nodes: 40 edges each = 150,000 edges
# - 12500 medium nodes: 3 edges each = 37,500 edges
# - 33750 low nodes: 0-1 edges = ~12,500 edges
# Total: ~200,000 edges

hub_ids = list(range(3750))
medium_ids = list(range(3750, 16250))
low_ids = list(range(16250, 50000))

def gen_targets_local(source_id, degree, num_nodes, seed_offset):
    """Generate targets with 70% locality bias."""
    r = random.Random(source_id * 1000 + seed_offset)
    targets = set()
    attempts = 0
    while len(targets) < degree and attempts < degree * 20:
        if r.random() < 0.7:
            offset = r.randint(-500, 500)
            target = (source_id + offset) % num_nodes
        else:
            target = r.randint(0, num_nodes - 1)
        if target != source_id:
            targets.add(target)
        attempts += 1
    return sorted(targets)

all_edges = []

# Hub nodes: 40 edges each → 3750 * 40 = 150,000
for node in hub_ids:
    targets = gen_targets_local(node, 40, num_nodes, 1)
    for t in targets:
        all_edges.append((node, t))

# Medium nodes: 3 edges each → 12500 * 3 = 37,500
for node in medium_ids:
    targets = gen_targets_local(node, 3, num_nodes, 2)
    for t in targets:
        all_edges.append((node, t))

# Low nodes: generate ~12,500 edges total from 33,750 nodes
# ~37% of low nodes get 1 edge
for i, node in enumerate(low_ids):
    if i % 3 == 0:  # every 3rd low node gets 1 edge → 33750/3 = 11,250 edges
        targets = gen_targets_local(node, 1, num_nodes, 3)
        for t in targets:
            all_edges.append((node, t))

# Deduplicate
unique_edges = list(set(all_edges))
print(f"Generated {len(unique_edges)} unique edges")

with open(output_path, 'w') as f:
    for src, tgt in unique_edges:
        f.write(f"{src} {tgt}\n")

print(f"Written to {output_path}")
