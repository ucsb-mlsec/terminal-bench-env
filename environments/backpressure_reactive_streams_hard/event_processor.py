#!/usr/bin/env python3

import asyncio
import json
import sys
import time
from typing import List, Dict, Any

# Global list that accumulates all events (MEMORY LEAK)
processed_events_cache = []
all_tasks = []

async def transform_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Transform a single event with async processing."""
    # Simulate async work
    await asyncio.sleep(0.001)
    
    transformed = {
        'id': event['id'],
        'original_value': event['value'],
        'value': event['value'] * 2,
        'timestamp': event['timestamp'],
        'processing_time': time.time()
    }
    
    # Store in global cache (MEMORY LEAK - never cleared)
    processed_events_cache.append(transformed)
    
    return transformed

async def process_event_pipeline(event: Dict[str, Any]) -> Dict[str, Any]:
    """Process event through multiple stages."""
    stage1 = await transform_event(event)
    
    # Additional transformation
    await asyncio.sleep(0.0005)
    stage1['processed'] = True
    
    return stage1

async def process_batch(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process a batch of events asynchronously."""
    # Create tasks but don't properly manage them (BUG)
    tasks = []
    for event in events:
        task = asyncio.create_task(process_event_pipeline(event))
        tasks.append(task)
        all_tasks.append(task)  # Keep reference forever (MEMORY LEAK)
    
    # Return immediately without awaiting (BUG - tasks accumulate)
    results = []
    for task in tasks:
        result = await task
        results.append(result)
    
    return results

def load_events(input_file: str) -> List[Dict[str, Any]]:
    """Load ALL events into memory at once (MEMORY ISSUE)."""
    events = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events

def write_events(output_file: str, events: List[Dict[str, Any]]):
    """Buffer all output in memory before writing (MEMORY ISSUE)."""
    # Build entire output in memory first
    output_buffer = []
    for event in events:
        output_buffer.append(json.dumps(event))
    
    # Write all at once
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_buffer))
        f.write('\n')

async def main():
    if len(sys.argv) != 3:
        print("Usage: python event_processor.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    start_time = time.time()
    
    # Load ALL events into memory (BUG)
    print("Loading events...")
    events = load_events(input_file)
    print(f"Loaded {len(events)} events")
    
    # Process events in small batches but accumulate results (BUG)
    all_results = []
    batch_size = 100
    
    for i in range(0, len(events), batch_size):
        batch = events[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}...")
        
        # Create new event loop for each batch (ANTI-PATTERN)
        results = await process_batch(batch)
        
        # Accumulate all results in memory (MEMORY LEAK)
        all_results.extend(results)
        
        # Keep reference to original events too (MEMORY LEAK)
        for event in batch:
            event['_processed'] = True
    
    print(f"Writing {len(all_results)} events...")
    write_events(output_file, all_results)
    
    elapsed = time.time() - start_time
    print(f"Processed {len(all_results)} events in {elapsed:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())