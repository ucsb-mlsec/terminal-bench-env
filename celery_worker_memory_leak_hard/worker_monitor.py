#!/usr/bin/env python3

import datetime
import collections
import threading

# Global module-level list that stores all metrics indefinitely (memory leak)
metrics_history = []

class MetricsCollector:
    def __init__(self):
        # Dictionary that stores task start times but never removes completed tasks (memory leak)
        self.task_timings = {}
        self.lock = threading.Lock()
    
    def record_task_start(self, task_id):
        """Record when a task starts - never removes completed tasks (memory leak)"""
        with self.lock:
            self.task_timings[task_id] = datetime.datetime.now()
    
    def record_task_end(self, task_id):
        """Calculate task duration and append to metrics_history"""
        with self.lock:
            if task_id in self.task_timings:
                start_time = self.task_timings[task_id]
                end_time = datetime.datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                # Append to global metrics_history list (memory leak)
                metrics_history.append({
                    'task_id': task_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration
                })
                # Note: task_id is NOT removed from self.task_timings (memory leak)
    
    def get_stats(self):
        """Return metrics without clearing old data"""
        with self.lock:
            return {
                'total_metrics': len(metrics_history),
                'active_tasks': len(self.task_timings),
                'history': metrics_history[-100:]  # Only return last 100 but doesn't clear
            }

# Global module-level MetricsCollector instance
collector = MetricsCollector()

def log_task_execution(task_name, duration):
    """Log task execution metrics to global metrics_history list"""
    metrics_dict = {
        'timestamp': datetime.datetime.now(),
        'task_name': task_name,
        'duration': duration,
        'thread_id': threading.current_thread().ident
    }
    
    # Append to global list without any cleanup or rotation (memory leak)
    metrics_history.append(metrics_dict)
    
    # No cleanup, no rotation, no eviction policy implemented
