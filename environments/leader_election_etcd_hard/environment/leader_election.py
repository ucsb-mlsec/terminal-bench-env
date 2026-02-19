from etcd_client import EtcdClient

class LeaderElector:
    def __init__(self, etcd_client, instance_id, lease_ttl=10):
        """Initialize the leader elector with etcd client, unique instance ID, and lease TTL"""
        self.etcd_client = etcd_client
        self.instance_id = instance_id
        self.lease_ttl = lease_ttl
        self.leader_key = '/election/leader'
        self.lease_id = None
        self.is_leader = False
    
    def attempt_election(self):
        """Try to become the leader - BUGGY: has race conditions and improper lease handling"""
        # BUG: Non-atomic check-then-set pattern creates race condition
        current_leader = self.etcd_client.get(self.leader_key)
        
        if current_leader is None:
            # BUG: Create lease but don't properly bind it to the key atomically
            self.lease_id = self.etcd_client.create_lease(self.lease_ttl)
            
            # BUG: This is a separate operation, not atomic with the check above
            # Multiple instances can pass the check and all try to set
            self.etcd_client.put(self.leader_key, self.instance_id, lease=self.lease_id)
            
            self.is_leader = True
            return True
        
        # BUG: If key exists, just give up without proper handling
        return False
    
    def is_leader_active(self):
        """Check if this instance is currently the leader - BUGGY: doesn't verify against etcd"""
        # BUG: Just returns local state without checking if we still hold the key in etcd
        # Doesn't detect if lease expired or another instance took over
        return self.is_leader
    
    def get_current_leader(self):
        """Query etcd to see who the current leader is"""
        leader_value = self.etcd_client.get(self.leader_key)
        return leader_value
    
    def resign(self):
        """Voluntarily give up leadership - BUGGY: doesn't clean up lease"""
        # BUG: Delete the key but don't revoke the lease
        self.etcd_client.delete(self.leader_key)
        
        # BUG: Set local state to False but leave the lease active
        # This can cause issues as the lease is still consuming resources
        self.is_leader = False
        
        return True
    
    def renew_leadership(self):
        """Keep the leadership lease alive - BUGGY: doesn't verify still leader"""
        if self.lease_id is None:
            return False
        
        # BUG: Tries to renew but doesn't check if we're still actually the leader in etcd
        # Another instance might have taken over, but we'd keep renewing our orphaned lease
        try:
            result = self.etcd_client.keep_alive(self.lease_id)
            return result
        except:
            return False