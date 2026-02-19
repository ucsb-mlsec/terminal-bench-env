#!/usr/bin/env python3

import random
import logging
from typing import Dict, List, Tuple, Optional, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MemorySimulator:
    """Simulates concurrent memory allocation patterns across multiple streams."""
    
    def __init__(self, allocator, num_streams: int = 3, operations_per_stream: int = 1000):
        """
        Initialize the memory simulator.
        
        Args:
            allocator: Memory allocator instance with allocate() and free() methods
            num_streams: Number of concurrent processing streams
            operations_per_stream: Number of operations each stream will perform
        """
        self.allocator = allocator
        self.num_streams = num_streams
        self.operations_per_stream = operations_per_stream
        self.workload: List[Tuple[str, int, Dict[str, Any]]] = []
        self.stream_allocations: Dict[int, List[int]] = {i: [] for i in range(num_streams)}
        self.next_block_id = 0
        
    def generate_workload(self) -> List[Tuple[str, int, Dict[str, Any]]]:
        """
        Generate a workload of allocation and free operations.
        
        Returns:
            List of tuples: (operation_type, stream_id, params)
            operation_type: 'allocate' or 'free'
            params: {'size_mb': int} for allocate, {'block_id': int} for free
        """
        random.seed(42)
        workload = []
        
        # Track potential blocks to free per stream
        stream_blocks: Dict[int, List[int]] = {i: [] for i in range(self.num_streams)}
        block_counter = 0
        
        # Generate operations for each stream
        for stream_id in range(self.num_streams):
            for op_idx in range(self.operations_per_stream):
                # 70% allocate, 30% free (but only if we have blocks to free)
                if random.random() < 0.7 or len(stream_blocks[stream_id]) == 0:
                    # Allocate operation
                    size_mb = random.randint(1, 50)
                    workload.append(('allocate', stream_id, {'size_mb': size_mb, 'block_id': block_counter}))
                    stream_blocks[stream_id].append(block_counter)
                    block_counter += 1
                else:
                    # Free operation - pick a random allocated block
                    block_to_free = random.choice(stream_blocks[stream_id])
                    stream_blocks[stream_id].remove(block_to_free)
                    workload.append(('free', stream_id, {'block_id': block_to_free}))
        
        # Interleave operations from different streams to simulate concurrency
        # while maintaining ordering within each stream
        stream_positions = [0] * self.num_streams
        operations_per_stream_list = [
            [op for op in workload if op[1] == sid] 
            for sid in range(self.num_streams)
        ]
        
        interleaved_workload = []
        total_ops = sum(len(ops) for ops in operations_per_stream_list)
        
        while len(interleaved_workload) < total_ops:
            # Round-robin across streams
            for stream_id in range(self.num_streams):
                if stream_positions[stream_id] < len(operations_per_stream_list[stream_id]):
                    interleaved_workload.append(operations_per_stream_list[stream_id][stream_positions[stream_id]])
                    stream_positions[stream_id] += 1
        
        self.workload = interleaved_workload
        logger.info(f"Generated workload with {len(self.workload)} operations across {self.num_streams} streams")
        return self.workload
    
    def run_simulation(self) -> Dict[str, Any]:
        """
        Execute the simulation workload.
        
        Returns:
            Dict with keys:
                - success: bool
                - total_operations: int
                - failed_allocations: int
                - completed_operations: int
        """
        if not self.workload:
            self.generate_workload()
        
        failed_allocations = 0
        completed_operations = 0
        
        # Track active allocations per stream with their sizes
        active_allocations: Dict[int, Dict[int, int]] = {i: {} for i in range(self.num_streams)}
        
        logger.info("Starting simulation...")
        
        for idx, (op_type, stream_id, params) in enumerate(self.workload):
            if idx > 0 and idx % 100 == 0:
                logger.info(f"Progress: {idx}/{len(self.workload)} operations, "
                          f"failed: {failed_allocations}, "
                          f"active blocks: {sum(len(blocks) for blocks in active_allocations.values())}")
            
            if op_type == 'allocate':
                size_mb = params['size_mb']
                block_id = params['block_id']
                
                result = self.allocator.allocate(stream_id, size_mb)
                
                if result is None:
                    failed_allocations += 1
                    logger.warning(f"Allocation failed: stream={stream_id}, size={size_mb}MB, "
                                 f"operation={idx}")
                    
                    # Stop if too many failures
                    if failed_allocations > 10:
                        logger.error("Too many allocation failures, stopping simulation")
                        break
                else:
                    active_allocations[stream_id][block_id] = size_mb
                    completed_operations += 1
                    
            elif op_type == 'free':
                block_id = params['block_id']
                
                # Only free if it was successfully allocated
                if block_id in active_allocations[stream_id]:
                    self.allocator.free(stream_id, block_id)
                    del active_allocations[stream_id][block_id]
                    completed_operations += 1
                else:
                    # Block was never allocated (previous allocation failed)
                    completed_operations += 1
        
        success = (failed_allocations == 0 and completed_operations == len(self.workload))
        
        results = {
            'success': success,
            'total_operations': len(self.workload),
            'failed_allocations': failed_allocations,
            'completed_operations': completed_operations
        }
        
        logger.info(f"Simulation complete: {results}")
        return results


class NaiveAllocator:
    """Naive memory allocator without pooling - causes fragmentation."""
    
    def __init__(self, total_memory_mb: int = 2048):
        self.total_memory_mb = total_memory_mb
        self.used_memory_mb = 0
        self.allocations: Dict[Tuple[int, int], int] = {}  # (stream_id, block_id) -> size
    
    def allocate(self, stream_id: int, size_mb: int) -> Optional[int]:
        """Allocate memory for a stream."""
        if self.used_memory_mb + size_mb > self.total_memory_mb:
            return None
        
        block_id = len(self.allocations)
        self.allocations[(stream_id, block_id)] = size_mb
        self.used_memory_mb += size_mb
        return block_id
    
    def free(self, stream_id: int, block_id: int) -> None:
        """Free allocated memory."""
        key = (stream_id, block_id)
        if key in self.allocations:
            size = self.allocations[key]
            self.used_memory_mb -= size
            del self.allocations[key]


if __name__ == '__main__':
    # Test with naive allocator
    logger.info("Testing with NaiveAllocator...")
    naive_allocator = NaiveAllocator(total_memory_mb=2048)
    simulator = MemorySimulator(naive_allocator, num_streams=3, operations_per_stream=1000)
    results = simulator.run_simulation()
    
    print("\n" + "="*60)
    print("NAIVE ALLOCATOR RESULTS:")
    print(f"Success: {results['success']}")
    print(f"Total Operations: {results['total_operations']}")
    print(f"Failed Allocations: {results['failed_allocations']}")
    print(f"Completed Operations: {results['completed_operations']}")
    print("="*60)