#!/usr/bin/env python3

"""
Task Processing Simulator
A deterministic simulator for testing parallel task scheduling algorithms.
"""

class Task:
    """Represents a computational task with an ID and duration."""
    
    def __init__(self, task_id, duration):
        self.id = task_id
        self.duration = duration
    
    def __repr__(self):
        return f"Task(id={self.id}, duration={self.duration})"


class Worker:
    """Represents a worker that processes tasks."""
    
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.current_task = None
        self.time_remaining = 0
        self.is_idle = True
        self.completed_tasks = []
    
    def assign_task(self, task):
        """Assign a new task to this worker."""
        if task is None:
            return
        self.current_task = task
        self.time_remaining = task.duration
        self.is_idle = False
    
    def tick(self):
        """Simulate one time unit of work."""
        if self.current_task is not None and self.time_remaining > 0:
            self.time_remaining -= 1
            if self.time_remaining == 0:
                # Task completed
                self.completed_tasks.append(self.current_task)
                self.current_task = None
                self.is_idle = True
                return True  # Signal task completion
        return False
    
    def is_busy(self):
        """Check if worker is currently processing a task."""
        return not self.is_idle
    
    def __repr__(self):
        return f"Worker(id={self.worker_id}, idle={self.is_idle}, task={self.current_task})"


class Simulator:
    """
    Deterministic simulator for parallel task processing.
    Runs step-by-step simulation without actual threading.
    """
    
    def __init__(self, scheduler, workload):
        self.scheduler = scheduler
        self.workload = workload
        self.workers = []
        self.tasks = []
        self.current_time = 0
        
        # Metrics
        self.tasks_completed = 0
        self.idle_violations = 0
        self.tasks_stolen = 0
        self.initial_task_count = 0
        
        self.load_workload(workload)
    
    def load_workload(self, config):
        """Parse and load workload configuration."""
        num_workers = config.get('num_workers', 4)
        tasks_config = config.get('tasks', [])
        
        # Create workers
        self.workers = [Worker(i) for i in range(num_workers)]
        
        # Create tasks
        self.tasks = []
        for task_config in tasks_config:
            task_id = task_config.get('id')
            duration = task_config.get('duration', 1)
            self.tasks.append(Task(task_id, duration))
        
        self.initial_task_count = len(self.tasks)
        
        # Initialize scheduler with workers
        self.scheduler.initialize(self.workers)
    
    def run(self):
        """
        Main simulation loop.
        Processes tasks step-by-step until all tasks are completed.
        """
        # Phase 1: Initial task assignment
        self.scheduler.assign_tasks(self.tasks)
        
        # Phase 2: Simulation loop
        max_iterations = 1000000  # Safety limit
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            all_idle = True
            
            # Process each worker for this time step
            for worker in self.workers:
                if worker.is_busy():
                    all_idle = False
                    task_completed = worker.tick()
                    
                    if task_completed:
                        self.tasks_completed += 1
                        # Notify scheduler that worker is now idle
                        self.scheduler.on_worker_idle(worker)
            
            # Check for idle violations BEFORE getting next tasks
            violations = self.detect_idle_violations()
            self.idle_violations += violations
            
            # Assign tasks to idle workers
            for worker in self.workers:
                if worker.is_idle:
                    # Track if this is a stolen task
                    initial_queue_state = self._get_queue_snapshot()
                    
                    next_task = self.scheduler.get_next_task(worker)
                    
                    if next_task is not None:
                        # Check if task was stolen from another worker
                        if self._was_task_stolen(next_task, worker, initial_queue_state):
                            self.tasks_stolen += 1
                        
                        worker.assign_task(next_task)
                        all_idle = False
            
            # Check if simulation is complete
            if all_idle:
                break
            
            self.current_time += 1
        
        return self.tasks_completed == self.initial_task_count
    
    def detect_idle_violations(self):
        """
        Detect if any worker is idle while other workers have tasks.
        Returns the number of idle violations in this time step.
        """
        violations = 0
        
        # Check if any worker has pending tasks
        any_tasks_remaining = self.scheduler.has_tasks_remaining()
        
        if any_tasks_remaining:
            # Count idle workers
            for worker in self.workers:
                if worker.is_idle:
                    violations += 1
        
        return violations
    
    def _get_queue_snapshot(self):
        """Get current state of task queues for each worker."""
        snapshot = {}
        if hasattr(self.scheduler, 'worker_queues'):
            for worker_id, queue in self.scheduler.worker_queues.items():
                snapshot[worker_id] = list(queue)
        return snapshot
    
    def _was_task_stolen(self, task, receiving_worker, initial_state):
        """
        Check if a task was stolen (taken from another worker's queue).
        """
        if not hasattr(self.scheduler, 'worker_queues'):
            return False
        
        # Check if task came from a different worker's queue
        for worker_id, initial_queue in initial_state.items():
            if worker_id == receiving_worker.worker_id:
                continue
            
            # Check if this task was in another worker's queue
            if any(t.id == task.id for t in initial_queue):
                # Verify it's no longer in that queue
                current_queue = self.scheduler.worker_queues.get(worker_id, [])
                if not any(t.id == task.id for t in current_queue):
                    return True
        
        return False
    
    def get_metrics(self):
        """Return simulation metrics."""
        return {
            'tasks_completed': self.tasks_completed,
            'idle_violations': self.idle_violations,
            'tasks_stolen': self.tasks_stolen,
            'total_time': self.current_time
        }


if __name__ == "__main__":
    # Simple test of the simulator
    from collections import deque
    
    class SimpleScheduler:
        """Basic round-robin scheduler for testing."""
        
        def initialize(self, workers):
            self.workers = workers
            self.worker_queues = {w.worker_id: deque() for w in workers}
        
        def assign_tasks(self, tasks):
            # Round-robin assignment
            for i, task in enumerate(tasks):
                worker_id = i % len(self.workers)
                self.worker_queues[worker_id].append(task)
        
        def on_worker_idle(self, worker):
            pass
        
        def get_next_task(self, worker):
            queue = self.worker_queues[worker.worker_id]
            if queue:
                return queue.popleft()
            return None
        
        def has_tasks_remaining(self):
            return any(len(q) > 0 for q in self.worker_queues.values())
    
    # Test workload
    workload = {
        'num_workers': 2,
        'tasks': [
            {'id': i, 'duration': 5} for i in range(10)
        ]
    }
    
    scheduler = SimpleScheduler()
    sim = Simulator(scheduler, workload)
    success = sim.run()
    metrics = sim.get_metrics()
    
    print(f"Simulation completed: {success}")
    print(f"Metrics: {metrics}")