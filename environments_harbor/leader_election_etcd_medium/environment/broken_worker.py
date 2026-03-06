#!/usr/bin/env python3

import etcd3
import time
import sys
import socket
import uuid
import signal

class DistributedWorker:
    def __init__(self, etcd_host='localhost', etcd_port=2379):
        self.worker_id = str(uuid.uuid4())
        self.hostname = socket.gethostname()
        self.etcd_client = etcd3.client(host=etcd_host, port=etcd_port)
        self.is_leader = False
        self.should_stop = False
        self.leader_key = '/leader'
        self.lease = None
        
        print(f"Worker {self.worker_id} on {self.hostname} initialized")
    
    def acquire_leadership(self):
        """Attempt to acquire leadership - BUGGY IMPLEMENTATION"""
        try:
            # BUG 1: Not checking if leader already exists
            # BUG 2: Not using lease with proper TTL
            # BUG 3: Not using atomic compare-and-swap
            
            # Just write the leader key without any checks
            self.etcd_client.put(self.leader_key, self.worker_id)
            self.is_leader = True
            print(f"Worker {self.worker_id} claimed leadership!")
            return True
            
        except Exception as e:
            print(f"Worker {self.worker_id} failed to acquire leadership: {e}")
            return False
    
    def maintain_leadership(self):
        """Maintain leadership by refreshing lease - BUGGY IMPLEMENTATION"""
        try:
            # BUG 1: No actual lease refresh happening
            # BUG 2: Not checking if we're still the leader
            # BUG 3: Missing error handling for lease expiration
            
            if self.is_leader:
                # This doesn't actually maintain anything
                pass
                
        except Exception as e:
            print(f"Worker {self.worker_id} error maintaining leadership: {e}")
            self.is_leader = False
    
    def release_leadership(self):
        """Release leadership when stopping"""
        if self.is_leader:
            try:
                leader_value = self.etcd_client.get(self.leader_key)[0]
                if leader_value and leader_value.decode('utf-8') == self.worker_id:
                    self.etcd_client.delete(self.leader_key)
                    print(f"Worker {self.worker_id} released leadership")
            except Exception as e:
                print(f"Worker {self.worker_id} error releasing leadership: {e}")
            
            self.is_leader = False
    
    def process_tasks(self):
        """Simulate task processing"""
        if self.is_leader:
            print(f"[LEADER] Worker {self.worker_id} processing tasks...")
            time.sleep(2)
    
    def run(self):
        """Main worker loop"""
        print(f"Worker {self.worker_id} starting...")
        
        # Try to acquire leadership
        self.acquire_leadership()
        
        # Main processing loop
        while not self.should_stop:
            try:
                self.maintain_leadership()
                self.process_tasks()
                time.sleep(1)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Worker {self.worker_id} error in main loop: {e}")
                time.sleep(1)
        
        self.release_leadership()
        print(f"Worker {self.worker_id} stopped")
    
    def stop(self):
        """Signal the worker to stop"""
        self.should_stop = True

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\nReceived shutdown signal")
    if 'worker' in globals():
        worker.stop()
    sys.exit(0)

if __name__ == '__main__':
    worker = DistributedWorker()
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    worker.run()