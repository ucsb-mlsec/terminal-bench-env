#!/usr/bin/env python3

import threading
from concurrent.futures import ThreadPoolExecutor
import time


class RedisSimulator:
    """In-memory Redis simulator for testing"""
    
    def __init__(self):
        self.data = {}
        self.scripts = {}
        self.lock = threading.Lock()
    
    def set(self, key, value):
        with self.lock:
            self.data[key] = str(value)
    
    def get(self, key):
        with self.lock:
            return self.data.get(key)
    
    def incr(self, key):
        with self.lock:
            current = int(self.data.get(key, 0))
            current += 1
            self.data[key] = str(current)
            return current
    
    def decr(self, key):
        with self.lock:
            current = int(self.data.get(key, 0))
            current -= 1
            self.data[key] = str(current)
            return current
    
    def register_script(self, script):
        script_id = str(len(self.scripts))
        self.scripts[script_id] = script
        
        class ScriptObject:
            def __init__(self, script_id, simulator):
                self.script_id = script_id
                self.simulator = simulator
            
            def evalsha(self, keys, args):
                return self.simulator._execute_lua(self.script_id, keys, args)
        
        return ScriptObject(script_id, self)
    
    def exists(self, key):
        with self.lock:
            return key in self.data
    
    def delete(self, key):
        with self.lock:
            if key in self.data:
                del self.data[key]
    
    def _execute_lua(self, script_id, keys, args):
        """Simulate Lua script execution with the BUGS"""
        # BUG 1: Non-atomic operations - check and decrement are separate
        time.sleep(0.001)  # Simulate small delay to exacerbate race condition
        
        inventory_key = keys[0]
        reservation_key = keys[1]
        quantity = int(args[0])
        customer_id = args[1]
        
        # BUG 1: Check happens BEFORE decrement (non-atomic)
        current_inventory = self.get(inventory_key)
        if current_inventory is None:
            current_inventory = 0
        else:
            current_inventory = int(current_inventory)
        
        # Race condition window here - multiple threads can pass this check
        if current_inventory >= quantity:
            # Small delay to make race condition more likely
            time.sleep(0.001)
            
            # BUG 3: Decrement happens but no rollback mechanism if reservation fails
            new_inventory = current_inventory - quantity
            self.set(inventory_key, new_inventory)
            
            # Simulate setting reservation record (could fail but no rollback)
            self.set(reservation_key, customer_id)
            
            # BUG 2: Returns 1 for success (should be consistent with expectation)
            return 1
        else:
            # BUG 2: Returns 0 for failure
            return 0


class InventoryManager:
    """Manages inventory operations with Redis and Lua scripts"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        
        # Lua script with THREE BUGS
        self.reserve_script_lua = """
        local inventory_key = KEYS[1]
        local reservation_key = KEYS[2]
        local quantity = tonumber(ARGV[1])
        local customer_id = ARGV[2]
        
        -- BUG 1: Non-atomic operation - check and decrement are separate
        local current = redis.call('GET', inventory_key)
        if current == false then
            current = 0
        else
            current = tonumber(current)
        end
        
        -- Race condition: multiple requests can pass this check
        if current >= quantity then
            -- BUG 3: No rollback if reservation record fails
            redis.call('DECRBY', inventory_key, quantity)
            redis.call('SET', reservation_key, customer_id)
            
            -- BUG 2: Returns 1 for success (inconsistent with some expectations)
            return 1
        else
            -- BUG 2: Returns 0 for failure
            return 0
        end
        """
        
        self.reserve_script = self.redis.register_script(self.reserve_script_lua)
    
    def reserve_item(self, product_id, customer_id, quantity=1):
        """Reserve inventory for a customer"""
        inventory_key = f'inventory:{product_id}'
        reservation_key = f'reservation:{product_id}:{customer_id}'
        
        result = self.reserve_script.evalsha(
            keys=[inventory_key, reservation_key],
            args=[quantity, customer_id]
        )
        
        # Returns True if result is 1 (success)
        return result == 1
    
    def get_inventory(self, product_id):
        """Get current inventory count"""
        inventory_key = f'inventory:{product_id}'
        value = self.redis.get(inventory_key)
        return int(value) if value is not None else 0
    
    def set_inventory(self, product_id, quantity):
        """Set initial inventory"""
        inventory_key = f'inventory:{product_id}'
        self.redis.set(inventory_key, quantity)


def test_concurrent_reservations():
    """Test concurrent reservations to demonstrate the bugs"""
    print("Starting concurrent reservation test...")
    
    # Create Redis simulator and inventory manager
    redis_sim = RedisSimulator()
    inventory_mgr = InventoryManager(redis_sim)
    
    # Set initial inventory to 10
    product_id = 'PRODUCT_001'
    initial_inventory = 10
    inventory_mgr.set_inventory(product_id, initial_inventory)
    
    print(f"Initial inventory: {initial_inventory}")
    
    # Track successful reservations
    successful_reservations = []
    lock = threading.Lock()
    
    def attempt_reservation(customer_num):
        customer_id = f'CUSTOMER_{customer_num}'
        success = inventory_mgr.reserve_item(product_id, customer_id, quantity=1)
        if success:
            with lock:
                successful_reservations.append(customer_id)
        return success
    
    # Spawn 50 concurrent reservation attempts
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(attempt_reservation, i) for i in range(50)]
        results = [f.result() for f in futures]
    
    # Count successful reservations
    success_count = len(successful_reservations)
    final_inventory = inventory_mgr.get_inventory(product_id)
    
    print(f"Expected: {initial_inventory}, Actual: {success_count}")
    print(f"Final inventory: {final_inventory}")
    print(f"Overselling detected: {success_count > initial_inventory}")
    
    return success_count


if __name__ == "__main__":
    result = test_concurrent_reservations()