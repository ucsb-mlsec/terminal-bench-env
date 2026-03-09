#!/usr/bin/env python3
"""Script to fix ppi_network.txt by removing LLM commentary and adding more edges."""

import random

# Read current file, skip first 2 lines (commentary + blank)
with open('/wekafs/kazhu/code/terminal-bench-env/environments_harbor/igraph_community_detection_medium/environment/data/ppi_network.txt', 'r') as f:
    lines = f.readlines()

# The first line is LLM commentary, second is blank
# Valid data starts at line 3 (index 2)
valid_lines = []
for line in lines[2:]:
    line = line.strip()
    if line and '\t' in line:
        parts = line.split('\t')
        if len(parts) == 2:
            valid_lines.append(line)

print(f"Valid edges found: {len(valid_lines)}")

# Extract all existing edges to avoid duplicates
existing_edges = set()
for line in valid_lines:
    p1, p2 = line.split('\t')
    edge = (min(p1, p2), max(p1, p2))
    existing_edges.add(edge)

print(f"Unique edges: {len(existing_edges)}")

# Extract all protein IDs
all_proteins = set()
for edge in existing_edges:
    all_proteins.add(edge[0])
    all_proteins.add(edge[1])

print(f"Total proteins: {len(all_proteins)}")

# Generate additional edges to reach ~2000
random.seed(42)
protein_list = sorted(all_proteins)
additional_edges = []

# Need about 766 more edges
target = 2000
attempts = 0
while len(valid_lines) + len(additional_edges) < target and attempts < 100000:
    attempts += 1
    p1 = random.choice(protein_list)
    p2 = random.choice(protein_list)
    if p1 == p2:
        continue
    edge = (min(p1, p2), max(p1, p2))
    if edge not in existing_edges:
        existing_edges.add(edge)
        additional_edges.append(f"{edge[0]}\t{edge[1]}")

print(f"Additional edges generated: {len(additional_edges)}")
print(f"Total edges: {len(valid_lines) + len(additional_edges)}")

# Write the fixed file
output_lines = valid_lines + additional_edges
with open('/wekafs/kazhu/code/terminal-bench-env/environments_harbor/igraph_community_detection_medium/environment/data/ppi_network.txt', 'w') as f:
    f.write('\n'.join(output_lines) + '\n')

print("Done! File written.")
