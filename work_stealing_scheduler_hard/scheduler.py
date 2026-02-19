#!/usr/bin/env python3

from collections import deque

class Scheduler:
    """
    Broken task scheduler that assigns tasks upfront without load balancing.
    Workers that finish early remain idle while others are overloaded.
    """
    
    def __init__(self, num_workers):
        """
        Initialize the scheduler with a number of workers.
        
        Args:
            num_workers: Number of worker threads to manage
        """
        self.num_workers = num_workers
        self.worker_queues = [deque() for _ in range(num_workers)]
        self.current_worker = 0
    
    def assign_tasks(self, tasks):
        """
        Initial task assignment - uses simple round-robin distribution.
        This is broken because it doesn't consider task duration or complexity.
        
        Args:
            tasks: List of task objects with 'id' and 'duration' fields
        """
        # Broken implementation: simple round-robin without load balancing
        for task in tasks:
            self.worker_queues[self.current_worker].append(task)
            self.current_worker = (self.current_worker + 1) % self.num_workers
    
    def on_worker_idle(self, worker_id):
        """
        Called when a worker becomes idle (finishes all assigned tasks).
        
        BROKEN: This method does nothing, leaving workers idle while
        other workers still have tasks in their queues.
        
        Args:
            worker_id: The ID of the worker that became idle
        """
        # Broken: does nothing - no work stealing implemented
        pass
    
    def get_next_task(self, worker_id):
        """
        Get the next task for a specific worker.
        
        Args:
            worker_id: The ID of the worker requesting a task
            
        Returns:
            Next task object from the worker's queue, or None if queue is empty
        """
        if worker_id < 0 or worker_id >= self.num_workers:
            return None
        
        if self.worker_queues[worker_id]:
            return self.worker_queues[worker_id].popleft()
        return None
    
    def get_queue_size(self, worker_id):
        """
        Get the current queue size for a worker.
        
        Args:
            worker_id: The ID of the worker
            
        Returns:
            Number of tasks in the worker's queue
        """
        if worker_id < 0 or worker_id >= self.num_workers:
            return 0
        return len(self.worker_queues[worker_id])
    
    def has_tasks(self):
        """
        Check if any worker still has tasks.
        
        Returns:
            True if any worker queue is non-empty, False otherwise
        """
        return any(len(queue) > 0 for queue in self.worker_queues)