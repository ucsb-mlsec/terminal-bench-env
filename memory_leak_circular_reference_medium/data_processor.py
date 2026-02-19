#!/usr/bin/env python3

import time
import gc
import os
import psutil

class DataBatch:
    """Represents a batch of data to be processed"""
    def __init__(self, batch_id, data):
        self.batch_id = batch_id
        self.data = data
        self.size = len(data)
        self.processor_ref = None  # Will store reference to processor
    
    def get_summary(self):
        return f"Batch {self.batch_id}: {self.size} items"


class ProcessorState:
    """Maintains processing state and history"""
    def __init__(self, processor):
        self.processor = processor
        self.processed_batches = []
        self.stats = {'total': 0, 'success': 0}
    
    def record_batch(self, batch):
        self.processed_batches.append(batch)
        self.stats['total'] += 1
        self.stats['success'] += 1


class Processor:
    """Main data processor with state management"""
    def __init__(self):
        self.state = ProcessorState(self)
        self.batch_count = 0
        self.results = []
    
    def process_batch(self, batch):
        """Process a single batch of data"""
        # Create circular reference: batch -> processor
        batch.processor_ref = self
        
        # Simulate processing
        result = sum(batch.data)
        self.batch_count += 1
        
        # Store batch in state (maintains reference)
        self.state.record_batch(batch)
        
        self.results.append({
            'batch_id': batch.batch_id,
            'result': result,
            'summary': batch.get_summary()
        })
        
        return result


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def main():
    processor = Processor()
    
    print("Starting data processor...")
    print(f"Initial memory: {get_memory_usage():.2f} MB")
    
    # Process batches in a loop
    for i in range(100):
        # Create a new batch with some data
        data = list(range(1000))  # 1000 integers per batch
        batch = DataBatch(batch_id=i, data=data)
        
        # Process the batch
        processor.process_batch(batch)
        
        # Periodic memory check
        if (i + 1) % 10 == 0:
            gc.collect()  # Force garbage collection
            mem = get_memory_usage()
            print(f"Processed {i + 1} batches, Memory: {mem:.2f} MB")
        
        time.sleep(0.01)  # Small delay to simulate real processing
    
    print(f"\nFinal memory: {get_memory_usage():.2f} MB")
    print(f"Total batches processed: {processor.batch_count}")


if __name__ == "__main__":
    main()