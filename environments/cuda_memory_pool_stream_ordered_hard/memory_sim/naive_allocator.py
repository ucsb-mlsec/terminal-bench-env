#!/usr/bin/env python3
"""
Naive Memory Allocator - Demonstrates fragmentation issues
This allocator uses simple linear allocation without pooling,
causing severe fragmentation after ~500 operations per stream.
"""


class NaiveAllocator:
    """
    A naive memory allocator that demonstrates fragmentation problems.
    Uses first-fit allocation strategy without any pooling mechanism.
    """
    
    def __init__(self, total_memory_mb):
        """
        Initialize the allocator with total available memory.
        
        Args:
            total_memory_mb: Total memory capacity in megabytes
        """
        self.total_memory = total_memory_mb
        self.used_memory = 0
        self.next_block_id = 0
        
        # Track allocated blocks: block_id -> (stream_id, size_mb, offset)
        self.allocated_blocks = {}
        
        # Track free regions: list of (offset, size) tuples
        self.free_regions = [(0, total_memory_mb)]
        
        # Statistics
        self.failed_allocations = 0
        self.total_allocations = 0
        
    def allocate(self, stream_id, size_mb):
        """
        Allocate a memory block of given size for a stream.
        Uses first-fit strategy - finds first free region large enough.
        
        Args:
            stream_id: ID of the stream requesting memory
            size_mb: Size of memory block in megabytes
            
        Returns:
            Unique block_id on success, None on failure
        """
        self.total_allocations += 1
        
        # First-fit: find first free region that can accommodate the request
        for i, (offset, region_size) in enumerate(self.free_regions):
            if region_size >= size_mb:
                # Found a suitable region
                block_id = self.next_block_id
                self.next_block_id += 1
                
                # Record the allocation
                self.allocated_blocks[block_id] = (stream_id, size_mb, offset)
                self.used_memory += size_mb
                
                # Update free regions
                if region_size == size_mb:
                    # Exact fit - remove this free region
                    self.free_regions.pop(i)
                else:
                    # Partial fit - shrink the free region
                    self.free_regions[i] = (offset + size_mb, region_size - size_mb)
                
                return block_id
        
        # No suitable free region found - allocation failed
        self.failed_allocations += 1
        return None
    
    def free(self, stream_id, block_id):
        """
        Free a previously allocated memory block.
        Attempts to merge adjacent free regions to reduce fragmentation.
        
        Args:
            stream_id: ID of the stream freeing memory
            block_id: ID of the block to free
        """
        if block_id not in self.allocated_blocks:
            return
        
        alloc_stream_id, size_mb, offset = self.allocated_blocks[block_id]
        
        # Remove from allocated blocks
        del self.allocated_blocks[block_id]
        self.used_memory -= size_mb
        
        # Insert the freed region back into free list
        self._insert_and_merge_free_region(offset, size_mb)
    
    def _insert_and_merge_free_region(self, offset, size):
        """
        Insert a freed region and merge with adjacent free regions.
        This helps reduce fragmentation but doesn't eliminate it entirely.
        
        Args:
            offset: Starting offset of the freed region
            size: Size of the freed region
        """
        # Find where to insert this free region (sorted by offset)
        insert_pos = 0
        for i, (free_offset, _) in enumerate(self.free_regions):
            if offset < free_offset:
                insert_pos = i
                break
            insert_pos = i + 1
        
        # Insert the new free region
        self.free_regions.insert(insert_pos, (offset, size))
        
        # Merge adjacent free regions
        self._merge_adjacent_regions()
    
    def _merge_adjacent_regions(self):
        """
        Merge adjacent free regions to create larger contiguous blocks.
        """
        if len(self.free_regions) <= 1:
            return
        
        merged = []
        current_offset, current_size = self.free_regions[0]
        
        for i in range(1, len(self.free_regions)):
            next_offset, next_size = self.free_regions[i]
            
            # Check if current and next regions are adjacent
            if current_offset + current_size == next_offset:
                # Merge them
                current_size += next_size
            else:
                # Not adjacent, save current and move to next
                merged.append((current_offset, current_size))
                current_offset, current_size = next_offset, next_size
        
        # Add the last region
        merged.append((current_offset, current_size))
        
        self.free_regions = merged
    
    def get_stats(self):
        """
        Get current allocator statistics.
        
        Returns:
            Dictionary with memory usage statistics
        """
        return {
            'total_memory': self.total_memory,
            'used_memory': self.used_memory,
            'free_memory': self.total_memory - self.used_memory,
            'num_blocks': len(self.allocated_blocks),
            'num_free_regions': len(self.free_regions),
            'failed_allocations': self.failed_allocations,
            'total_allocations': self.total_allocations
        }