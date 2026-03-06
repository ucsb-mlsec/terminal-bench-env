#!/usr/bin/env python3

import configparser
import os
import sys

CONFIG_DIR = '/opt/jobscheduler/config/'
errors = []

def validate_server_conf():
    """Validate server.conf file"""
    config = configparser.ConfigParser()
    filepath = os.path.join(CONFIG_DIR, 'server.conf')
    
    try:
        config.read(filepath)
    except configparser.Error as e:
        errors.append(f"ERROR in server.conf: Syntax error - {str(e)}")
        return
    
    # Check for required fields
    required_fields = ['primary_server', 'backup_server', 'failover_timeout', 'heartbeat_interval']
    
    if 'failover' not in config.sections():
        errors.append("ERROR in server.conf: Missing [failover] section")
        return
    
    for field in required_fields:
        if field not in config['failover']:
            errors.append(f"ERROR in server.conf: Missing required field '{field}'")
    
    # Check for negative timeout values
    if 'failover_timeout' in config['failover']:
        try:
            timeout = int(config['failover']['failover_timeout'])
            if timeout < 0:
                errors.append("ERROR in server.conf: failover_timeout cannot be negative")
        except ValueError:
            errors.append("ERROR in server.conf: failover_timeout must be an integer")
    
    if 'heartbeat_interval' in config['failover']:
        try:
            interval = int(config['failover']['heartbeat_interval'])
            if interval < 0:
                errors.append("ERROR in server.conf: heartbeat_interval cannot be negative")
        except ValueError:
            errors.append("ERROR in server.conf: heartbeat_interval must be an integer")

def validate_resources_conf():
    """Validate resources.conf file"""
    config = configparser.ConfigParser()
    filepath = os.path.join(CONFIG_DIR, 'resources.conf')
    
    try:
        config.read(filepath)
    except configparser.Error as e:
        errors.append(f"ERROR in resources.conf: Syntax error - {str(e)}")
        return
    
    # Check for required fields
    required_fields = ['max_cpu_cores', 'max_memory_gb', 'max_concurrent_jobs']
    
    if 'limits' not in config.sections():
        errors.append("ERROR in resources.conf: Missing [limits] section")
        return
    
    for field in required_fields:
        if field not in config['limits']:
            errors.append(f"ERROR in resources.conf: Missing required field '{field}'")
    
    # Check for negative values
    if 'max_cpu_cores' in config['limits']:
        try:
            cores = int(config['limits']['max_cpu_cores'])
            if cores < 0:
                errors.append("ERROR in resources.conf: max_cpu_cores cannot be negative")
        except ValueError:
            errors.append("ERROR in resources.conf: max_cpu_cores must be an integer")
    
    if 'max_memory_gb' in config['limits']:
        try:
            memory = int(config['limits']['max_memory_gb'])
            if memory < 0:
                errors.append("ERROR in resources.conf: max_memory_gb cannot be negative")
        except ValueError:
            errors.append("ERROR in resources.conf: max_memory_gb must be an integer")
    
    if 'max_concurrent_jobs' in config['limits']:
        try:
            jobs = int(config['limits']['max_concurrent_jobs'])
            if jobs < 0:
                errors.append("ERROR in resources.conf: max_concurrent_jobs cannot be negative")
        except ValueError:
            errors.append("ERROR in resources.conf: max_concurrent_jobs must be an integer")

def validate_priority_conf():
    """Validate priority.conf file"""
    config = configparser.ConfigParser()
    filepath = os.path.join(CONFIG_DIR, 'priority.conf')
    
    try:
        config.read(filepath)
    except configparser.Error as e:
        errors.append(f"ERROR in priority.conf: Syntax error - {str(e)}")
        return
    
    # Check for required priority levels
    required_levels = ['LOW', 'MEDIUM', 'HIGH']
    
    if 'priorities' not in config.sections():
        errors.append("ERROR in priority.conf: Missing [priorities] section")
        return
    
    for level in required_levels:
        if level not in config['priorities']:
            errors.append(f"ERROR in priority.conf: Missing required priority level '{level}'")
    
    # Check priority values are in 0-10 range
    for level in ['LOW', 'MEDIUM', 'HIGH']:
        if level in config['priorities']:
            try:
                value = int(config['priorities'][level])
                if value < 0 or value > 10:
                    errors.append(f"ERROR in priority.conf: {level} priority value {value} exceeds maximum of 10" if value > 10 else f"ERROR in priority.conf: {level} priority value cannot be negative")
            except ValueError:
                errors.append(f"ERROR in priority.conf: {level} priority value must be an integer")

def main():
    # Check if config directory exists
    if not os.path.exists(CONFIG_DIR):
        print(f"ERROR: Configuration directory {CONFIG_DIR} does not exist")
        sys.exit(1)
    
    # Validate each configuration file
    validate_server_conf()
    validate_resources_conf()
    validate_priority_conf()
    
    # Report results
    if errors:
        for error in errors:
            print(error)
        sys.exit(1)
    else:
        print("All configurations valid")
        sys.exit(0)

if __name__ == "__main__":
    main()