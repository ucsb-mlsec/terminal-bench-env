#!/usr/bin/env python3
"""Generate social_graph.txt with realistic social network properties.

The graph has:
- 50,000 nodes (IDs 0-49999)
- ~200,000 directed edges
- Locality: 70% of edges connect nearby node IDs
- Power-law degree distribution: hub/medium/low node tiers
"""
import random

random.seed(42)

NUM_NODES = 50000
OUTPUT = '/data/social_graph.txt'


def gen_targets(source_id, degree, num_nodes, rng):
    """Generate `degree` distinct target node IDs with locality bias."""
    targets = set()
    attempts = 0
    while len(targets) < degree and attempts < degree * 20:
        if rng.random() < 0.7:
            offset = rng.randint(-500, 500)
            target = (source_id + offset) % num_nodes
        else:
            target = rng.randint(0, num_nodes - 1)
        if target != source_id:
            targets.add(target)
        attempts += 1
    return targets


all_edges = set()
rng = random.Random(42)

# Hub nodes (0-3749): ~40 edges each → 3750 * 40 = 150,000
for node in range(3750):
    targets = gen_targets(node, 40, NUM_NODES, rng)
    for t in targets:
        all_edges.add((node, t))

# Medium nodes (3750-16249): 3 edges each → 12,500 * 3 = 37,500
for node in range(3750, 16250):
    targets = gen_targets(node, 3, NUM_NODES, rng)
    for t in targets:
        all_edges.add((node, t))

# Low nodes (16250-49999): every 3rd node gets 1 edge → ~11,250
for node in range(16250, NUM_NODES):
    if node % 3 == 0:
        targets = gen_targets(node, 1, NUM_NODES, rng)
        for t in targets:
            all_edges.add((node, t))

print(f"Generated {len(all_edges)} unique edges")

with open(OUTPUT, 'w') as f:
    for src, tgt in all_edges:
        f.write(f"{src} {tgt}\n")

print(f"Written to {OUTPUT}")
