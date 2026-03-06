#!/usr/bin/env python3

import threading
import re


class MockRedis:
    def __init__(self):
        self.data = {}
        self.lock = threading.Lock()
    
    def get(self, key):
        with self.lock:
            return self.data.get(key)
    
    def set(self, key, value):
        with self.lock:
            self.data[key] = str(value)
    
    def incr(self, key, amount=1):
        with self.lock:
            current = self.data.get(key, "0")
            new_val = int(current) + amount
            self.data[key] = str(new_val)
            return new_val
    
    def incrby(self, key, amount):
        return self.incr(key, amount)
    
    def eval(self, script, numkeys, *args):
        with self.lock:
            # Parse KEYS and ARGV from args
            keys = list(args[:numkeys])
            argv = list(args[numkeys:])
            
            # Simple Lua script interpreter for common patterns
            # Support for increment pattern: get, add, set, return
            
            # Create a context for script execution
            result = None
            
            # Parse and execute the script
            # Handle common pattern: get current value, increment, set, return
            if "redis.call" in script:
                # Extract redis.call operations
                lines = script.split(';')
                local_vars = {}
                
                for line in lines:
                    line = line.strip()
                    
                    # Handle local variable assignments with redis.call("get", ...)
                    if 'local' in line and '=' in line and 'redis.call' in line:
                        var_match = re.search(r'local\s+(\w+)\s*=', line)
                        if var_match:
                            var_name = var_match.group(1)
                            
                            if '"get"' in line or "'get'" in line:
                                # Extract key reference
                                if 'KEYS[1]' in line:
                                    key = keys[0] if keys else None
                                    if key:
                                        val = self.data.get(key, "0")
                                        # Handle 'or "0"' pattern
                                        if 'or' in line:
                                            val = val if val is not None else "0"
                                        local_vars[var_name] = val
                            
                            # Handle arithmetic operations
                            elif 'tonumber' in line and '+' in line:
                                # Parse expression like: tonumber(current) + tonumber(ARGV[1])
                                parts = line.split('=', 1)[1].strip()
                                # Extract variables and perform calculation
                                if '+' in parts:
                                    # Simple addition
                                    operands = re.findall(r'tonumber\((\w+|\w+\[\d+\])\)', parts)
                                    total = 0
                                    for op in operands:
                                        if op in local_vars:
                                            total += int(local_vars[op])
                                        elif op.startswith('ARGV'):
                                            idx = int(re.search(r'\[(\d+)\]', op).group(1)) - 1
                                            total += int(argv[idx])
                                    local_vars[var_name] = total
                    
                    # Handle redis.call("set", ...)
                    elif 'redis.call' in line and ('"set"' in line or "'set'" in line):
                        # Extract key and value
                        if 'KEYS[1]' in line:
                            key = keys[0] if keys else None
                            # Find value being set
                            for var_name, var_val in local_vars.items():
                                if var_name in line:
                                    if key:
                                        self.data[key] = str(var_val)
                                    break
                    
                    # Handle return statement
                    elif 'return' in line:
                        ret_match = re.search(r'return\s+(\w+)', line)
                        if ret_match:
                            var_name = ret_match.group(1)
                            if var_name in local_vars:
                                result = local_vars[var_name]
            
            return result