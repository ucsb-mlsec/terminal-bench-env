# Game Tree Structure and Selection Algorithm

## Overview
This directory contains data for a simplified Go game AI using Monte Carlo Tree Search (MCTS). The system builds a game tree where each node represents a board state, and uses simulation-based evaluation to determine optimal moves.

## Tree Structure

### Node Components
Each node in the tree represents a specific board state and contains:
- **Node ID**: Unique identifier (e.g., "n0", "n1", "n2")
- **Visits**: Number of simulations that have passed through this node
- **Wins**: Number of simulations through this node that resulted in victories
- **Children**: List of child node IDs representing possible next moves
- **Parent**: Reference to the parent node (except for root)

### Tree Hierarchy
- The root node is identified as **'n0'**
- Child nodes represent different move options from the current state
- The tree grows as more simulations explore different game continuations
- Each path from root to leaf represents a sequence of moves

## Selection Algorithm: UCB1 (Upper Confidence Bound)

The system uses the UCB1 formula to balance exploitation and exploration when selecting which child node to simulate next.

### Formula
```
UCB_score = (wins / visits) + c * sqrt(ln(parent_visits) / visits)
```

### Components Explained

**Exploitation Term: (wins / visits)**
- Represents the empirical win rate of this node
- Higher values indicate moves that have historically performed well
- Pure exploitation would always select the highest win rate

**Exploration Term: c * sqrt(ln(parent_visits) / visits)**
- Encourages exploring less-visited nodes
- Grows with parent visits (more simulations = more confidence = more exploration)
- Decreases with node visits (well-tested nodes need less exploration)
- The sqrt(ln(...)) function provides diminishing exploration pressure

**Exploration Parameter: c**
- Controls the exploration vs exploitation tradeoff
- Higher c: more exploration, more willing to try uncertain options
- Lower c: more exploitation, prefer known good moves
- Typical values range from 1.0 to 2.0
- Read from config.txt for this implementation

### Selection Process
1. Calculate UCB score for each child of the current node
2. Select the child with the highest score
3. If selected child is a leaf, run simulation
4. If selected child has children, recurse (select again from its children)
5. Backpropagate results: update visits and wins for all nodes in the path

## Edge Cases

### Unvisited Nodes (visits = 0)
Nodes that have never been visited present a mathematical challenge:
- The formula includes division by visits (undefined for 0)
- The exploration term becomes infinity
- **Solution**: Unvisited nodes should receive maximum priority

**Implementation approaches:**
- Return a very high score (e.g., infinity or 1e9)
- Handle separately by always selecting unvisited nodes first
- Ensures all children are tried at least once before deeper exploration

### Numerical Stability
- Use natural logarithm (ln) as specified in UCB1
- Handle floating point precision appropriately
- Round final scores to 3 decimal places for output

## Data Format

### tree_data.json
Contains the complete game tree structure:
```json
{
  "node_id": {
    "visits": <integer>,
    "wins": <integer>,
    "children": [<list of child node IDs>],
    "parent": "<parent node ID or null for root>"
  }
}
```

### config.txt
Contains the exploration parameter:
```
c=<float value>
```

### Output: scores.json
Your implementation should calculate UCB scores for all direct children of node 'n0':
```json
{
  "node_id": <score rounded to 3 decimals>,
  "node_id": <score rounded to 3 decimals>
}
```

## Task Requirements

1. Read the tree structure from tree_data.json
2. Read the exploration parameter c from config.txt
3. Identify all children of the root node 'n0'
4. For each child, calculate its UCB score using:
   - The child's wins and visits
   - The parent's (n0's) visits
   - The exploration parameter c
5. Handle unvisited nodes appropriately (assign very high scores)
6. Round scores to 3 decimal places
7. Save results to /workspace/solution/scores.json

## Monte Carlo Tree Search Context

This selection algorithm is a core component of MCTS, which consists of four phases:
1. **Selection**: Use UCB1 to traverse tree to a leaf (this task)
2. **Expansion**: Add new child nodes to grow the tree
3. **Simulation**: Play out a random game from the new node
4. **Backpropagation**: Update statistics along the path

The UCB1 formula ensures that the search efficiently balances trying promising moves with exploring uncertain areas of the game tree, leading to strong play even with limited computational resources.