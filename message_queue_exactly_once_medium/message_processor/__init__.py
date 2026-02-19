#!/usr/bin/env python3

import json
import os
import hashlib
from pathlib import Path

class MessageProcessor:
    def __init__(self, state_file='/workspace/message_processor/processed_ids.dat'):
        self.state_file = state_file
        self.processed_ids = self._load_processed_ids()
        self.output_file = '/workspace/output/transactions.txt'
        self.duplicate_count = 0
        self.unique_processed = 0
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
    def _load_processed_ids(self):
        """Load previously processed message IDs from state file"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return set(line.strip() for line in f if line.strip())
        return set()
    
    def _save_processed_id(self, message_id):
        """Persist a processed message ID to state file"""
        with open(self.state_file, 'a') as f:
            f.write(f"{message_id}\n")
    
    def _get_message_id(self, message):
        """Extract or generate a unique message ID"""
        # First try to get explicit message ID
        if 'message_id' in message:
            return str(message['message_id'])
        elif 'id' in message:
            return str(message['id'])
        
        # If no ID field, create hash from message content
        content = json.dumps(message, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def process_message(self, message):
        """Process a single message with duplicate detection"""
        message_id = self._get_message_id(message)
        
        # Check if already processed
        if message_id in self.processed_ids:
            self.duplicate_count += 1
            return False  # Duplicate, skip processing
        
        # Process the transaction
        self._process_transaction(message)
        
        # Mark as processed
        self.processed_ids.add(message_id)
        self._save_processed_id(message_id)
        self.unique_processed += 1
        
        return True  # Successfully processed
    
    def _process_transaction(self, message):
        """Process the payment transaction and write to output file"""
        transaction_record = self._create_transaction_record(message)
        with open(self.output_file, 'a') as f:
            f.write(transaction_record + '\n')
    
    def _create_transaction_record(self, message):
        """Create a transaction record from message"""
        return json.dumps(message)
    
    def process_queue(self, queue_file):
        """Process all messages from the queue file"""
        with open(queue_file, 'r') as f:
            messages = json.load(f)
        
        if isinstance(messages, dict) and 'messages' in messages:
            messages = messages['messages']
        
        for message in messages:
            self.process_message(message)
    
    def get_stats(self):
        """Return processing statistics"""
        return {
            'duplicate_count': self.duplicate_count,
            'unique_processed': self.unique_processed,
            'state_file': self.state_file
        }


def main():
    # Initialize processor
    processor = MessageProcessor()
    
    # Process test queue
    test_queue = '/workspace/test_queue/messages.json'
    if os.path.exists(test_queue):
        processor.process_queue(test_queue)
    
    # Get statistics
    stats = processor.get_stats()
    
    # Write solution summary
    solution_content = f"""DUPLICATE_COUNT={stats['duplicate_count']}
UNIQUE_PROCESSED={stats['unique_processed']}
STATE_FILE={stats['state_file']}
"""
    
    with open('/workspace/solution.txt', 'w') as f:
        f.write(solution_content)
    
    print(f"Processing complete:")
    print(f"  Duplicates detected: {stats['duplicate_count']}")
    print(f"  Unique messages processed: {stats['unique_processed']}")
    print(f"  State file: {stats['state_file']}")


if __name__ == '__main__':
    main()