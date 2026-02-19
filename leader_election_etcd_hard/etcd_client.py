#!/usr/bin/env python3

import threading
import time
from typing import Any, Dict, List, Optional, Tuple


class EtcdClient:
    """
    A simulated etcd client for testing leader election.
    Implements core etcd features: KV storage, leases, and atomic transactions.
    """
    
    def __init__(self, host='localhost', port=2379):
        self.host = host
        self.port = port
        self.lock = threading.RLock()
        
        # Key-value store
        self.store: Dict[str, Dict[str, Any]] = {}
        
        # Lease management
        self.leases: Dict[int, Dict[str, Any]] = {}
        self.lease_counter = 0
        
        # Version tracking for compare-and-swap
        self.version_counter = 0
    
    def put(self, key: str, value: str, lease_id: Optional[int] = None) -> bool:
        """Store a key-value pair, optionally bound to a lease."""
        with self.lock:
            self._expire_leases()
            
            if lease_id is not None:
                if lease_id not in self.leases:
                    return False
                if self._is_lease_expired(lease_id):
                    return False
            
            self.version_counter += 1
            
            if key in self.store:
                self.store[key]['value'] = value
                self.store[key]['version'] = self.version_counter
                self.store[key]['lease_id'] = lease_id
            else:
                self.store[key] = {
                    'value': value,
                    'version': self.version_counter,
                    'create_revision': self.version_counter,
                    'lease_id': lease_id
                }
            
            return True
    
    def get(self, key: str) -> Optional[str]:
        """Retrieve the value for a key."""
        with self.lock:
            self._expire_leases()
            
            if key in self.store:
                return self.store[key]['value']
            return None
    
    def delete(self, key: str) -> bool:
        """Delete a key."""
        with self.lock:
            if key in self.store:
                del self.store[key]
                return True
            return False
    
    def grant_lease(self, ttl: int) -> int:
        """Create a lease with TTL in seconds."""
        with self.lock:
            self.lease_counter += 1
            lease_id = self.lease_counter
            
            self.leases[lease_id] = {
                'ttl': ttl,
                'created_at': time.time(),
                'last_keepalive': time.time()
            }
            
            return lease_id
    
    def revoke_lease(self, lease_id: int) -> bool:
        """Revoke a lease and delete all keys bound to it."""
        with self.lock:
            if lease_id not in self.leases:
                return False
            
            # Delete all keys bound to this lease
            keys_to_delete = [
                key for key, data in self.store.items()
                if data.get('lease_id') == lease_id
            ]
            for key in keys_to_delete:
                del self.store[key]
            
            del self.leases[lease_id]
            return True
    
    def keep_alive(self, lease_id: int) -> bool:
        """Refresh/renew a lease."""
        with self.lock:
            if lease_id not in self.leases:
                return False
            
            if self._is_lease_expired(lease_id):
                return False
            
            self.leases[lease_id]['last_keepalive'] = time.time()
            return True
    
    def transaction(self, compare_list: List[Tuple], success_ops: List[Tuple], 
                   failure_ops: List[Tuple]) -> Tuple[bool, Any]:
        """Execute an atomic compare-and-swap transaction."""
        with self.lock:
            self._expire_leases()
            
            # Evaluate all comparisons
            all_match = True
            for compare in compare_list:
                key = compare[0]
                compare_type = compare[1]
                expected = compare[2]
                
                if compare_type == 'create':
                    # Check if key doesn't exist (create_revision == 0 means not exists)
                    if expected == 0:
                        if key in self.store:
                            all_match = False
                            break
                    else:
                        if key not in self.store:
                            all_match = False
                            break
                        if self.store[key]['create_revision'] != expected:
                            all_match = False
                            break
                
                elif compare_type == 'version':
                    if key not in self.store:
                        all_match = False
                        break
                    if self.store[key]['version'] != expected:
                        all_match = False
                        break
                
                elif compare_type == 'value':
                    if key not in self.store:
                        if expected is not None:
                            all_match = False
                            break
                    else:
                        if self.store[key]['value'] != expected:
                            all_match = False
                            break
            
            # Execute appropriate operations
            ops_to_execute = success_ops if all_match else failure_ops
            results = []
            
            for op in ops_to_execute:
                op_type = op[0]
                
                if op_type == 'put':
                    key = op[1]
                    value = op[2]
                    lease_id = op[3] if len(op) > 3 else None
                    result = self.put(key, value, lease_id)
                    results.append(result)
                
                elif op_type == 'delete':
                    key = op[1]
                    result = self.delete(key)
                    results.append(result)
                
                elif op_type == 'get':
                    key = op[1]
                    result = self.get(key)
                    results.append(result)
            
            return (all_match, results)
    
    def _is_lease_expired(self, lease_id: int) -> bool:
        """Check if a lease has expired."""
        if lease_id not in self.leases:
            return True
        
        lease = self.leases[lease_id]
        elapsed = time.time() - lease['last_keepalive']
        return elapsed > lease['ttl']
    
    def _expire_leases(self):
        """Clean up expired leases and their keys."""
        expired_leases = []
        
        for lease_id, lease in self.leases.items():
            if self._is_lease_expired(lease_id):
                expired_leases.append(lease_id)
        
        for lease_id in expired_leases:
            self.revoke_lease(lease_id)
```

Now, here's the fixed leader election implementation:

```python
#!/usr/bin/env python3

import time
import threading
import uuid
from typing import Optional


class LeaderElector:
    """
    Implements distributed leader election using etcd.
    Ensures mutual exclusion through lease-based coordination.
    """
    
    def __init__(self, etcd_client, election_key='/election/leader', 
                 lease_ttl=10, instance_id=None):
        """
        Initialize the leader elector.
        
        Args:
            etcd_client: EtcdClient instance for coordination
            election_key: Key used for leader election
            lease_ttl: Lease TTL in seconds
            instance_id: Unique identifier for this instance
        """
        self.etcd = etcd_client
        self.election_key = election_key
        self.lease_ttl = lease_ttl
        self.instance_id = instance_id or str(uuid.uuid4())
        
        self.lease_id: Optional[int] = None
        self.is_leader = False
        self.lock = threading.RLock()
        self.keepalive_thread: Optional[threading.Thread] = None
        self.keepalive_stop = threading.Event()
    
    def attempt_election(self) -> bool:
        """
        Attempt to become the leader.
        
        Returns:
            True if successfully elected as leader, False otherwise
        """
        with self.lock:
            # Create a new lease
            lease_id = self.etcd.grant_lease(self.lease_ttl)
            if lease_id is None:
                return False
            
            # Try to atomically create the leader key with our instance_id
            # Only succeeds if the key doesn't exist (create_revision == 0)
            success, _ = self.etcd.transaction(
                compare_list=[
                    (self.election_key, 'create', 0)
                ],
                success_ops=[
                    ('put', self.election_key, self.instance_id, lease_id)
                ],
                failure_ops=[]
            )
            
            if success:
                self.lease_id = lease_id
                self.is_leader = True
                self._start_keepalive()
                return True
            else:
                # Failed to acquire leadership, revoke the lease
                self.etcd.revoke_lease(lease_id)
                return False
    
    def check_leadership(self) -> bool:
        """
        Check if this instance is currently the leader.
        
        Returns:
            True if this instance is the leader, False otherwise
        """
        with self.lock:
            if not self.is_leader:
                return False
            
            # Verify we still hold leadership by checking the key value
            current_leader = self.etcd.get(self.election_key)
            
            if current_leader != self.instance_id:
                # Lost leadership
                self._cleanup_leadership()
                return False
            
            return True
    
    def get_current_leader(self) -> Optional[str]:
        """
        Get the instance ID of the current leader.
        
        Returns:
            The leader's instance_id, or None if no leader
        """
        return self.etcd.get(self.election_key)
    
    def resign(self) -> bool:
        """
        Voluntarily give up leadership.
        
        Returns:
            True if successfully resigned, False otherwise
        """
        with self.lock:
            if not self.is_leader:
                return False
            
            # Delete the leader key
            self.etcd.delete(self.election_key)
            
            # Cleanup
            self._cleanup_leadership()
            
            return True
    
    def _start_keepalive(self):
        """Start the lease keepalive background thread."""
        self.keepalive_stop.clear()
        self.keepalive_thread = threading.Thread(target=self._keepalive_loop, daemon=True)
        self.keepalive_thread.start()
    
    def _keepalive_loop(self):
        """Background thread to keep the lease alive."""
        while not self.keepalive_stop.is_set():
            time.sleep(self.lease_ttl / 3)  # Refresh at 1/3 of TTL
            
            with self.lock:
                if not self.is_leader or self.lease_id is None:
                    break
                
                success = self.etcd.keep_alive(self.lease_id)
                
                if not success:
                    # Failed to renew lease, lost leadership
                    self._cleanup_leadership()
                    break
    
    def _cleanup_leadership(self):
        """Internal cleanup when losing leadership."""
        self.is_leader = False
        
        # Stop keepalive thread
        self.keepalive_stop.set()
        
        # Revoke lease if we have one
        if self.lease_id is not None:
            try:
                self.etcd.revoke_lease(self.lease_id)
            except:
                pass
            self.lease_id = None
    
    def __del__(self):
        """Cleanup on deletion."""
        if self.is_leader:
            try:
                self.resign()
            except:
                pass