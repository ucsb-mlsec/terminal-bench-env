#!/usr/bin/env python3

import sys
import json
import os

class Node:
    def __init__(self, node_id, role):
        self.node_id = node_id
        self.role = role
        self.log = []
        self.commit_index = -1  # -1 means no entries committed yet
        self.match_index = {}  # Only used by leader
        self.current_term = 1
        
    def __repr__(self):
        return f"Node({self.node_id}, role={self.role}, log_len={len(self.log)}, commit_index={self.commit_index})"

class RaftSimulator:
    def __init__(self):
        self.leader = Node("leader", "leader")
        self.follower1 = Node("follower1", "follower")
        self.follower2 = Node("follower2", "follower")
        
        # Leader tracks replication progress
        self.leader.match_index["follower1"] = -1
        self.leader.match_index["follower2"] = -1
        
        self.nodes = {
            "leader": self.leader,
            "follower1": self.follower1,
            "follower2": self.follower2
        }
        
    def handle_append(self, key, value):
        """Leader receives a new entry to replicate"""
        command = f"SET {key} {value}"
        entry = {
            "term": self.leader.current_term,
            "command": command
        }
        
        # BUG: Only appending to leader, not replicating to followers
        self.leader.log.append(entry)
        print(f"APPEND: Added entry to leader log: {command}")
        
    def handle_heartbeat(self):
        """Leader sends heartbeat to followers"""
        print("HEARTBEAT: Sending heartbeat...")
        
        # BUG: Incomplete replication logic
        # Should copy all log entries from leader to followers
        for follower_id in ["follower1", "follower2"]:
            follower = self.nodes[follower_id]
            # Only copying if follower log is shorter, but not copying all entries
            if len(follower.log) < len(self.leader.log):
                # BUG: Only copies one entry at a time instead of all missing entries
                next_index = len(follower.log)
                if next_index < len(self.leader.log):
                    # Copy only one entry
                    follower.log.append(self.leader.log[next_index])
                    print(f"  Replicated 1 entry to {follower_id}")
                    
    def handle_commit(self):
        """Leader commits entries up to current index"""
        # Leader commits all entries in its log
        if len(self.leader.log) > 0:
            self.leader.commit_index = len(self.leader.log) - 1
            print(f"COMMIT: Leader committed up to index {self.leader.commit_index}")
            
            # BUG: Not updating followers' commit_index
            # BUG: Not updating match_index for followers
            
    def process_operations(self, filepath):
        """Process operations from file"""
        if not os.path.exists(filepath):
            print(f"Error: Operations file not found: {filepath}")
            return
            
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                parts = line.split()
                operation = parts[0]
                
                if operation == "APPEND":
                    if len(parts) >= 3:
                        key = parts[1]
                        value = ' '.join(parts[2:])
                        self.handle_append(key, value)
                elif operation == "HEARTBEAT":
                    self.handle_heartbeat()
                elif operation == "COMMIT":
                    self.handle_commit()
                else:
                    print(f"Unknown operation: {operation}")
                    
    def print_state(self):
        """Print final state of all nodes"""
        print("\n" + "="*60)
        print("FINAL STATE:")
        print("="*60)
        
        for node_id in ["leader", "follower1", "follower2"]:
            node = self.nodes[node_id]
            print(f"\n{node.node_id.upper()}:")
            print(f"  Commit Index: {node.commit_index}")
            print(f"  Log Length: {len(node.log)}")
            print(f"  Log Entries:")
            for i, entry in enumerate(node.log):
                committed = "✓" if i <= node.commit_index else "✗"
                print(f"    [{i}] {entry['command']} (term={entry['term']}) {committed}")
                
        print("\n" + "="*60)
        
        # Check for inconsistencies
        inconsistencies = []
        if self.leader.commit_index != self.follower1.commit_index:
            inconsistencies.append(f"Leader commit_index ({self.leader.commit_index}) != Follower1 commit_index ({self.follower1.commit_index})")
        if self.leader.commit_index != self.follower2.commit_index:
            inconsistencies.append(f"Leader commit_index ({self.leader.commit_index}) != Follower2 commit_index ({self.follower2.commit_index})")
        if len(self.leader.log) != len(self.follower1.log):
            inconsistencies.append(f"Leader log length ({len(self.leader.log)}) != Follower1 log length ({len(self.follower1.log)})")
        if len(self.leader.log) != len(self.follower2.log):
            inconsistencies.append(f"Leader log length ({len(self.leader.log)}) != Follower2 log length ({len(self.follower2.log)})")
            
        if inconsistencies:
            print("\nINCONSISTENCIES DETECTED:")
            for issue in inconsistencies:
                print(f"  - {issue}")
        else:
            print("\n✓ All nodes are consistent!")
            
        return inconsistencies

def main():
    simulator = RaftSimulator()
    
    operations_file = "/workspace/raft_sim/operations.txt"
    print(f"Processing operations from: {operations_file}\n")
    
    simulator.process_operations(operations_file)
    simulator.print_state()
    
    # Write solution file
    solution_file = "/workspace/solution.txt"
    try:
        with open(solution_file, 'w') as f:
            f.write(f"leader_commit_index={simulator.leader.commit_index}\n")
            f.write(f"follower1_commit_index={simulator.follower1.commit_index}\n")
            f.write(f"follower2_commit_index={simulator.follower2.commit_index}\n")
        print(f"\nSolution written to: {solution_file}")
    except Exception as e:
        print(f"\nError writing solution file: {e}")

if __name__ == "__main__":
    main()