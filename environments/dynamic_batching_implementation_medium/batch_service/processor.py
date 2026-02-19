#!/usr/bin/env python3

import time
import threading
from queue import Queue, Empty
from typing import Any, Dict, List, Tuple
from processor import process

class BatchServer:
    def __init__(self, max_batch_size: int = 20, max_wait_time_ms: int = 50):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time_ms / 1000.0  # Convert to seconds
        self.request_queue = Queue()
        self.running = False
        self.batch_thread = None
        self.stats = {
            'total_batches': 0,
            'total_requests': 0,
            'batch_sizes': []
        }
        self.lock = threading.Lock()
        
    def start(self):
        """Start the batch processing thread"""
        self.running = True
        self.batch_thread = threading.Thread(target=self._batch_worker, daemon=True)
        self.batch_thread.start()
        
    def stop(self):
        """Stop the batch processing thread"""
        self.running = False
        if self.batch_thread:
            self.batch_thread.join(timeout=1.0)
            
    def submit_request(self, request_data: Any) -> Any:
        """
        Submit a request for processing.
        Returns the processed result.
        """
        response_queue = Queue()
        self.request_queue.put((request_data, response_queue))
        return response_queue.get()
        
    def _batch_worker(self):
        """Worker thread that collects and processes batches"""
        while self.running:
            batch = []
            response_queues = []
            start_time = time.time()
            
            # Collect requests until timeout or max batch size
            while len(batch) < self.max_batch_size:
                elapsed = time.time() - start_time
                timeout = max(0.001, self.max_wait_time - elapsed)
                
                try:
                    request_data, response_queue = self.request_queue.get(timeout=timeout)
                    batch.append(request_data)
                    response_queues.append(response_queue)
                    
                    # If we've reached max batch size, process immediately
                    if len(batch) >= self.max_batch_size:
                        break
                        
                except Empty:
                    # Timeout reached, process whatever we have
                    break
                    
                # Check if we've exceeded max wait time
                if time.time() - start_time >= self.max_wait_time:
                    break
            
            # Process the batch if we have any requests
            if batch:
                self._process_batch(batch, response_queues)
                
    def _process_batch(self, batch: List[Any], response_queues: List[Queue]):
        """Process a batch of requests and return results to response queues"""
        # Process the batch
        results = process(batch)
        
        # Update statistics
        with self.lock:
            self.stats['total_batches'] += 1
            self.stats['total_requests'] += len(batch)
            self.stats['batch_sizes'].append(len(batch))
        
        # Send results back to waiting threads
        for result, response_queue in zip(results, response_queues):
            response_queue.put(result)
            
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        with self.lock:
            stats = self.stats.copy()
        return stats


# Global server instance
_server = None

def start_server(max_batch_size: int = 20, max_wait_time_ms: int = 50):
    """Initialize and start the batch server"""
    global _server
    _server = BatchServer(max_batch_size, max_wait_time_ms)
    _server.start()
    return _server

def stop_server():
    """Stop the batch server"""
    global _server
    if _server:
        _server.stop()
        _server = None

def handle_request(request_data: Any) -> Any:
    """Handle a single request through the batch server"""
    global _server
    if _server is None:
        raise RuntimeError("Server not started. Call start_server() first.")
    return _server.submit_request(request_data)

def get_stats() -> Dict:
    """Get server statistics"""
    global _server
    if _server is None:
        return {'total_batches': 0, 'total_requests': 0, 'batch_sizes': []}
    return _server.get_stats()


if __name__ == "__main__":
    # Simple test
    print("Starting batch server test...")
    server = start_server(max_batch_size=20, max_wait_time_ms=50)
    
    # Submit some test requests
    import concurrent.futures
    
    def submit_test_request(i):
        result = handle_request(i)
        return result
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(submit_test_request, i) for i in range(50)]
        results = [f.result() for f in futures]
    
    stats = get_stats()
    print(f"Processed {stats['total_requests']} requests in {stats['total_batches']} batches")
    if stats['batch_sizes']:
        avg_batch_size = sum(stats['batch_sizes']) / len(stats['batch_sizes'])
        print(f"Average batch size: {avg_batch_size:.1f}")
    
    stop_server()
    print("Test complete")