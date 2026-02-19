#!/usr/bin/env python3

import etcd3
import time
import sys
import os
import signal
import threading

class DistributedWorker:
    def __init__(self, worker_id, etcd_host='localhost', etcd_port=2379):
        self.worker_id = worker_id
        self.etcd_host = etcd_host
        self.etcd_port = etcd_port
        self.client = None
        self.lease = None
        self.is_leader = False
        self.running = True
        self.election_key = '/leader/election'
        self.lease_ttl = 10
        
    def connect(self):
        """Establish connection to etcd"""
        try:
            self.client = etcd3.client(host=self.etcd_host, port=self.etcd_port)
            return True
        except Exception as e:
            print(f"[{self.worker_id}] Failed to connect to etcd: {e}", file=sys.stderr)
            return False
    
    def try_acquire_leadership(self):
        """Attempt to acquire leadership using etcd lease and transaction"""
        try:
            # Create a lease with TTL
            self.lease = self.client.lease(self.lease_ttl)
            
            # Try to create the election key with our worker_id using a transaction
            # This ensures atomic check-and-set operation
            success, _ = self.client.transaction(
                compare=[
                    self.client.transactions.version(self.election_key) == 0
                ],
                success=[
                    self.client.transactions.put(self.election_key, self.worker_id, lease=self.lease)
                ],
                failure=[]
            )
            
            if success:
                self.is_leader = True
                print(f"[{self.worker_id}] LEADER: Acquired leadership")
                return True
            else:
                # Key already exists, someone else is leader
                return False
                
        except Exception as e:
            print(f"[{self.worker_id}] Error acquiring leadership: {e}", file=sys.stderr)
            return False
    
    def maintain_leadership(self):
        """Keep refreshing the lease to maintain leadership"""
        while self.running and self.is_leader:
            try:
                # Refresh the lease
                if self.lease:
                    self.lease.refresh()
                
                # Verify we still hold the leadership key
                value, _ = self.client.get(self.election_key)
                if value is None or value.decode('utf-8') != self.worker_id:
                    print(f"[{self.worker_id}] LEADER: Lost leadership")
                    self.is_leader = False
                    break
                    
                time.sleep(self.lease_ttl / 3)  # Refresh more frequently than TTL
                
            except Exception as e:
                print(f"[{self.worker_id}] Error maintaining leadership: {e}", file=sys.stderr)
                self.is_leader = False
                break
    
    def wait_for_leadership(self):
        """Wait and watch for leadership opportunity"""
        print(f"[{self.worker_id}] FOLLOWER: Waiting for leadership opportunity")
        
        # Watch the election key for changes
        events_iterator, cancel = self.client.watch(self.election_key)
        
        try:
            for event in events_iterator:
                if not self.running:
                    break
                    
                # Check if the key was deleted or expired
                if isinstance(event, etcd3.events.DeleteEvent):
                    print(f"[{self.worker_id}] FOLLOWER: Leadership available, attempting to acquire")
                    if self.try_acquire_leadership():
                        break
                    else:
                        # Someone else got it, continue watching
                        cancel()
                        return self.wait_for_leadership()
                        
        except Exception as e:
            print(f"[{self.worker_id}] Error watching for leadership: {e}", file=sys.stderr)
        finally:
            cancel()
    
    def process_tasks(self):
        """Simulate task processing (only when leader)"""
        while self.running and self.is_leader:
            print(f"[{self.worker_id}] LEADER: Processing tasks...")
            time.sleep(2)
    
    def release_leadership(self):
        """Clean up and release leadership"""
        if self.is_leader:
            try:
                # Delete the election key
                self.client.delete(self.election_key)
                print(f"[{self.worker_id}] LEADER: Released leadership")
            except Exception as e:
                print(f"[{self.worker_id}] Error releasing leadership: {e}", file=sys.stderr)
        
        if self.lease:
            try:
                self.lease.revoke()
            except:
                pass
        
        self.is_leader = False
    
    def run(self):
        """Main worker loop"""
        if not self.connect():
            print(f"[{self.worker_id}] Failed to start: cannot connect to etcd", file=sys.stderr)
            return
        
        print(f"[{self.worker_id}] Worker started")
        
        while self.running:
            # Try to become leader
            if self.try_acquire_leadership():
                # Start lease maintenance in background
                lease_thread = threading.Thread(target=self.maintain_leadership, daemon=True)
                lease_thread.start()
                
                # Process tasks while leader
                self.process_tasks()
                
                # Wait for lease thread to finish
                lease_thread.join(timeout=1)
            else:
                # Wait for leadership opportunity
                self.wait_for_leadership()
        
        # Cleanup
        self.release_leadership()
        print(f"[{self.worker_id}] Worker stopped")
    
    def stop(self):
        """Signal worker to stop"""
        self.running = False
        self.release_leadership()


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print(f"\n[{worker.worker_id}] Received shutdown signal")
    worker.stop()
    sys.exit(0)


if __name__ == "__main__":
    # Generate worker ID from PID if not provided
    worker_id = os.getenv('WORKER_ID', f'worker-{os.getpid()}')
    
    worker = DistributedWorker(worker_id)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        worker.run()
    except Exception as e:
        print(f"[{worker_id}] Fatal error: {e}", file=sys.stderr)
        worker.stop()
        sys.exit(1)