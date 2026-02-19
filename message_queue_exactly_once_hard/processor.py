#!/usr/bin/env python3

import json
import os
import shutil
from pathlib import Path

QUEUE_DIR = '/workspace/queue'
COMPLETED_DIR = '/workspace/completed'
STATE_FILE = '/workspace/state.json'

def load_state():
    """Load the state file containing processed payment IDs"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {'processed': []}

def save_state(state):
    """Save the state file"""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def process_payments():
    """Process payment messages from queue directory"""
    
    # BUG 1: Load state only once at startup - doesn't check for concurrent updates
    state = load_state()
    processed_ids = set(state.get('processed', []))
    
    # Create completed directory if it doesn't exist
    os.makedirs(COMPLETED_DIR, exist_ok=True)
    
    # BUG 2: List all files at once - doesn't handle new files arriving during processing
    queue_files = os.listdir(QUEUE_DIR)
    
    for filename in queue_files:
        if not filename.endswith('.json'):
            continue
            
        filepath = os.path.join(QUEUE_DIR, filename)
        
        # BUG 3: Check if payment is in the old state (loaded at startup)
        # This means if we run the processor twice, the second run has stale state
        
        # BUG 4: Read file BEFORE checking state thoroughly
        try:
            with open(filepath, 'r') as f:
                message = json.load(f)
        except FileNotFoundError:
            # File might have been processed already
            continue
            
        payment_id = message.get('payment_id')
        amount = message.get('amount')
        
        # BUG 5: Only check in-memory state, not actual completed directory
        if payment_id in processed_ids:
            print(f"Skipping already processed payment: {payment_id}")
            continue
        
        # Process the payment
        print(f"Processing payment {payment_id}: ${amount}")
        
        # BUG 6: Move file to completed BEFORE updating state
        # If crash happens after move but before state update, payment is lost from state
        completed_path = os.path.join(COMPLETED_DIR, filename)
        shutil.move(filepath, completed_path)
        
        # BUG 7: Update in-memory state but don't write to disk yet
        processed_ids.add(payment_id)
        
        # BUG 8: No atomic operations - partial updates possible
    
    # BUG 9: Write state only at the end - if crash happens during processing,
    # all state updates are lost
    state['processed'] = list(processed_ids)
    save_state(state)
    
    print(f"Processed {len(queue_files)} files")

if __name__ == '__main__':
    process_payments()