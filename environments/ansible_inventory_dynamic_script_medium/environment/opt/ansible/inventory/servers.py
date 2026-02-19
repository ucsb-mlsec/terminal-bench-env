#!/usr/bin/env python

import sys
import json

def read_servers():
    # Bug: Wrong file path
    with open('/opt/ansible/data/server.json', 'r') as f:
        data = json.load(f)
    return data

def build_inventory():
    servers = read_servers()
    
    inventory = {}
    
    # Bug: Not handling the servers data structure correctly
    for server in servers:
        group = server['group']
        
        # Bug: Not initializing group structure properly
        if group not in inventory:
            inventory[group] = []
        
        # Bug: Appending server object instead of just hostname
        inventory[group].append(server)
    
    # Bug: Missing '_meta' group with hostvars
    # Bug: Not returning proper group structure with 'hosts' key
    
    return inventory

def main():
    # Bug: Not checking for --list argument that Ansible requires
    
    inventory = build_inventory()
    
    # Bug: Using print with comma instead of proper JSON output
    print json.dumps(inventory, indent=2)

# Bug: Wrong condition check
if __name__ = "__main__":
    main()