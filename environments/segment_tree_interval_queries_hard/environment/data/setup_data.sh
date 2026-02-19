#!/bin/bash

# Generate /tmp/data_values.txt with 100,000 random integers between -1000 and 1000
echo "Generating data_values.txt..."
> /tmp/data_values.txt
for i in {1..100000}; do
    echo $((RANDOM % 2001 - 1000)) >> /tmp/data_values.txt
done

# Generate /tmp/queries.txt with 5,000 mixed queries
echo "Generating queries.txt..."
> /tmp/queries.txt

for i in {1..5000}; do
    # Randomly select query type (0-4 for 5 types)
    query_type=$((RANDOM % 5))
    
    case $query_type in
        0)  # SUM L R
            L=$((RANDOM % 100000))
            R=$((L + RANDOM % (100000 - L)))
            echo "SUM $L $R" >> /tmp/queries.txt
            ;;
        1)  # MIN L R
            L=$((RANDOM % 100000))
            R=$((L + RANDOM % (100000 - L)))
            echo "MIN $L $R" >> /tmp/queries.txt
            ;;
        2)  # MAX L R
            L=$((RANDOM % 100000))
            R=$((L + RANDOM % (100000 - L)))
            echo "MAX $L $R" >> /tmp/queries.txt
            ;;
        3)  # UPDATE I V
            I=$((RANDOM % 100000))
            V=$((RANDOM % 2001 - 1000))
            echo "UPDATE $I $V" >> /tmp/queries.txt
            ;;
        4)  # RANGE_COUNT L R X
            L=$((RANDOM % 100000))
            R=$((L + RANDOM % (100000 - L)))
            X=$((RANDOM % 2001 - 1000))
            echo "RANGE_COUNT $L $R $X" >> /tmp/queries.txt
            ;;
    esac
done

echo "Data generation complete!"
echo "Generated /tmp/data_values.txt with 100,000 values"
echo "Generated /tmp/queries.txt with 5,000 queries"
```

Now for the query processor at /tmp/query_processor.py:

```python
#!/usr/bin/env python3

import sys
from bisect import bisect_right

class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.tree_sum = [0] * (4 * self.n)
        self.tree_min = [0] * (4 * self.n)
        self.tree_max = [0] * (4 * self.n)
        self.data = data[:]
        if self.n > 0:
            self.build(0, 0, self.n - 1)
    
    def build(self, node, start, end):
        if start == end:
            self.tree_sum[node] = self.data[start]
            self.tree_min[node] = self.data[start]
            self.tree_max[node] = self.data[start]
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            self.build(left_child, start, mid)
            self.build(right_child, mid + 1, end)
            
            self.tree_sum[node] = self.tree_sum[left_child] + self.tree_sum[right_child]
            self.tree_min[node] = min(self.tree_min[left_child], self.tree_min[right_child])
            self.tree_max[node] = max(self.tree_max[left_child], self.tree_max[right_child])
    
    def update(self, idx, value):
        self.data[idx] = value
        self._update(0, 0, self.n - 1, idx, value)
    
    def _update(self, node, start, end, idx, value):
        if start == end:
            self.tree_sum[node] = value
            self.tree_min[node] = value
            self.tree_max[node] = value
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            if idx <= mid:
                self._update(left_child, start, mid, idx, value)
            else:
                self._update(right_child, mid + 1, end, idx, value)
            
            self.tree_sum[node] = self.tree_sum[left_child] + self.tree_sum[right_child]
            self.tree_min[node] = min(self.tree_min[left_child], self.tree_min[right_child])
            self.tree_max[node] = max(self.tree_max[left_child], self.tree_max[right_child])
    
    def query_sum(self, L, R):
        return self._query_sum(0, 0, self.n - 1, L, R)
    
    def _query_sum(self, node, start, end, L, R):
        if R < start or L > end:
            return 0
        if L <= start and end <= R:
            return self.tree_sum[node]
        
        mid = (start + end) // 2
        left_sum = self._query_sum(2 * node + 1, start, mid, L, R)
        right_sum = self._query_sum(2 * node + 2, mid + 1, end, L, R)
        return left_sum + right_sum
    
    def query_min(self, L, R):
        return self._query_min(0, 0, self.n - 1, L, R)
    
    def _query_min(self, node, start, end, L, R):
        if R < start or L > end:
            return float('inf')
        if L <= start and end <= R:
            return self.tree_min[node]
        
        mid = (start + end) // 2
        left_min = self._query_min(2 * node + 1, start, mid, L, R)
        right_min = self._query_min(2 * node + 2, mid + 1, end, L, R)
        return min(left_min, right_min)
    
    def query_max(self, L, R):
        return self._query_max(0, 0, self.n - 1, L, R)
    
    def _query_max(self, node, start, end, L, R):
        if R < start or L > end:
            return float('-inf')
        if L <= start and end <= R:
            return self.tree_max[node]
        
        mid = (start + end) // 2
        left_max = self._query_max(2 * node + 1, start, mid, L, R)
        right_max = self._query_max(2 * node + 2, mid + 1, end, L, R)
        return max(left_max, right_max)
    
    def range_count(self, L, R, X):
        count = 0
        for i in range(L, R + 1):
            if self.data[i] > X:
                count += 1
        return count

def main():
    # Read data
    with open('/tmp/data_values.txt', 'r') as f:
        data = [int(line.strip()) for line in f]
    
    # Build segment tree
    seg_tree = SegmentTree(data)
    
    # Process queries
    results = []
    with open('/tmp/queries.txt', 'r') as f:
        for line in f:
            parts = line.strip().split()
            query_type = parts[0]
            
            if query_type == 'SUM':
                L, R = int(parts[1]), int(parts[2])
                results.append(str(seg_tree.query_sum(L, R)))
            elif query_type == 'MIN':
                L, R = int(parts[1]), int(parts[2])
                results.append(str(seg_tree.query_min(L, R)))
            elif query_type == 'MAX':
                L, R = int(parts[1]), int(parts[2])
                results.append(str(seg_tree.query_max(L, R)))
            elif query_type == 'UPDATE':
                I, V = int(parts[1]), int(parts[2])
                seg_tree.update(I, V)
                results.append('OK')
            elif query_type == 'RANGE_COUNT':
                L, R, X = int(parts[1]), int(parts[2]), int(parts[3])
                results.append(str(seg_tree.range_count(L, R, X)))
    
    # Write results
    with open('/tmp/query_results.txt', 'w') as f:
        f.write('\n'.join(results) + '\n')

if __name__ == '__main__':
    main()