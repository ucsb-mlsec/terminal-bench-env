#!/usr/bin/env python3

import sys
import time
from node import Node

def main():
    print("Starting test...")
    
    # Message queue for simulating async communication
    message_queue = []
    
    # Create 3 nodes: 1 leader and 2 followers
    nodes = {
        'node0': Node(node_id='node0', is_leader=True, total_nodes=3),
        'node1': Node(node_id='node1', is_leader=False, total_nodes=3),
        'node2': Node(node_id='node2', is_leader=False, total_nodes=3)
    }
    
    leader = nodes['node0']
    
    # Submit a log entry to the leader
    print("Submitting log entry to leader")
    leader.append_entry({'term': 1, 'command': 'set x=5'})
    
    # Trigger replication to followers
    messages = leader.replicate_to_followers()
    message_queue.extend(messages)
    
    print("Processing messages...")
    
    # Process messages with timeout
    start_time = time.time()
    max_iterations = 100
    iteration = 0
    
    while iteration < max_iterations and (time.time() - start_time) < 2.0:
        iteration += 1
        
        # Process all messages in the queue
        while message_queue:
            msg = message_queue.pop(0)
            target_node_id = msg['to']
            target_node = nodes[target_node_id]
            
            if msg['type'] == 'append_entries':
                # Follower handles append_entries
                response = target_node.handle_append_entries(msg)
                if response:
                    message_queue.append(response)
            elif msg['type'] == 'append_response':
                # Leader handles append_response
                new_messages = leader.handle_append_response(msg)
                if new_messages:
                    message_queue.extend(new_messages)
        
        # Check if commit index has advanced
        if leader.commit_index >= 0:
            print(f"SUCCESS: Entry committed! Commit index: {leader.commit_index}")
            
            # Save solution
            with open('/workspace/solution.json', 'w') as f:
                f.write('{\n')
                f.write('  "bug_file": "raft_system/node.py",\n')
                f.write('  "bug_line_number": 0,\n')
                f.write('  "test_status": "pass"\n')
                f.write('}\n')
            
            sys.exit(0)
        
        time.sleep(0.01)
    
    print(f"FAILURE: Entry not committed after {iteration} iterations")
    print(f"Leader commit_index: {leader.commit_index}")
    print(f"Leader log: {leader.log}")
    
    # Save solution with failure status
    with open('/workspace/solution.json', 'w') as f:
        f.write('{\n')
        f.write('  "bug_file": "raft_system/node.py",\n')
        f.write('  "bug_line_number": 0,\n')
        f.write('  "test_status": "fail"\n')
        f.write('}\n')
    
    sys.exit(1)

if __name__ == '__main__':
    main()