#!/usr/bin/env python3

class Node:
    """
    A simplified Raft-like consensus node implementation.
    Supports leader and follower roles with log replication and commit logic.
    """
    
    def __init__(self, node_id, is_leader, total_nodes):
        """
        Initialize a node in the distributed system.
        
        Args:
            node_id: Unique identifier for this node
            is_leader: Boolean indicating if this node is the leader
            total_nodes: Total number of nodes in the cluster
        """
        self.node_id = node_id
        self.is_leader = is_leader
        self.total_nodes = total_nodes
        self.log = []  # List of log entries
        self.commit_index = -1  # Index of highest log entry known to be committed
        
        # Leader-specific state
        if self.is_leader:
            # next_index: for each server, index of the next log entry to send
            self.next_index = {}
            # match_index: for each server, index of highest log entry known to be replicated
            self.match_index = {}
            
            # Initialize tracking for all other nodes
            for i in range(total_nodes):
                if i != node_id:
                    self.next_index[i] = 0
                    self.match_index[i] = -1
        else:
            self.next_index = None
            self.match_index = None
    
    def append_entry(self, entry):
        """
        Leader accepts a new log entry from a client.
        
        Args:
            entry: Dictionary representing the log entry (e.g., {'term': 1, 'command': 'set x=5'})
        
        Returns:
            The index where the entry was appended
        """
        if not self.is_leader:
            raise Exception("Only leader can accept new entries")
        
        self.log.append(entry)
        return len(self.log) - 1
    
    def replicate_to_followers(self):
        """
        Leader generates replication messages to send to all followers.
        
        Returns:
            List of AppendEntries messages to be sent to followers
        """
        if not self.is_leader:
            return []
        
        messages = []
        
        for follower_id in self.next_index:
            # Determine which entries to send to this follower
            next_idx = self.next_index[follower_id]
            entries_to_send = self.log[next_idx:]
            
            if entries_to_send:
                msg = {
                    'type': 'append_entries',
                    'from': self.node_id,
                    'to': follower_id,
                    'entries': entries_to_send,
                    'prev_log_index': next_idx - 1,
                    'leader_commit': self.commit_index
                }
                messages.append(msg)
        
        return messages
    
    def handle_append_entries(self, msg):
        """
        Follower processes an AppendEntries message from the leader.
        
        Args:
            msg: AppendEntries message dictionary
        
        Returns:
            AppendResponse message to send back to leader
        """
        if self.is_leader:
            # Leaders don't process append_entries in this simplified model
            return None
        
        entries = msg['entries']
        prev_log_index = msg['prev_log_index']
        
        # Check if our log is consistent with the leader's
        if prev_log_index >= 0 and prev_log_index >= len(self.log):
            # We're missing entries, reject
            return {
                'type': 'append_response',
                'from': self.node_id,
                'to': msg['from'],
                'success': False,
                'match_index': len(self.log) - 1
            }
        
        # Append new entries
        start_index = prev_log_index + 1
        for i, entry in enumerate(entries):
            idx = start_index + i
            if idx < len(self.log):
                # Overwrite existing entry (in case of conflict)
                self.log[idx] = entry
            else:
                # Append new entry
                self.log.append(entry)
        
        # Update commit index if leader has committed more entries
        if msg['leader_commit'] > self.commit_index:
            self.commit_index = min(msg['leader_commit'], len(self.log) - 1)
        
        # Send success response
        return {
            'type': 'append_response',
            'from': self.node_id,
            'to': msg['from'],
            'success': True,
            'match_index': len(self.log) - 1
        }
    
    def handle_append_response(self, msg):
        """
        Leader processes an AppendResponse from a follower.
        Updates match_index and determines if entries can be committed.
        
        Args:
            msg: AppendResponse message dictionary
        """
        if not self.is_leader:
            return
        
        follower_id = msg['from']
        
        if msg['success']:
            # Update our tracking of what this follower has replicated
            self.match_index[follower_id] = msg['match_index']
            self.next_index[follower_id] = msg['match_index'] + 1
            
            # BUG: The commit logic here has an error
            # Check if we can advance the commit index
            # An entry can be committed if it's replicated on a majority of nodes
            for index in range(self.commit_index + 1, len(self.log)):
                # Count how many nodes have this entry (including leader itself)
                replication_count = 1  # Leader has it
                
                for fid, match_idx in self.match_index.items():
                    if match_idx > index:  # BUG: Should be >= not >
                        replication_count += 1
                
                # Check if we have a majority
                majority = (self.total_nodes + 1) // 2
                if replication_count >= majority:
                    self.commit_index = index
        else:
            # Follower rejected, decrement next_index and retry
            self.next_index[follower_id] = max(0, self.next_index[follower_id] - 1)
    
    def get_committed_entries(self):
        """
        Returns all log entries up to and including the commit index.
        
        Returns:
            List of committed log entries
        """
        if self.commit_index < 0:
            return []
        return self.log[:self.commit_index + 1]