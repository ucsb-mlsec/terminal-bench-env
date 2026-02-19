#!/usr/bin/env python3

import os
import json
from pathlib import Path
from collections import defaultdict, deque

def find_all_symlinks(root_path):
    """Find all symbolic links in the directory tree."""
    symlinks = []
    for dirpath, dirnames, filenames in os.walk(root_path, followlinks=False):
        # Check directories for symlinks
        for dirname in dirnames:
            full_path = os.path.join(dirpath, dirname)
            if os.path.islink(full_path):
                symlinks.append(full_path)
        # Check files for symlinks
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            if os.path.islink(full_path):
                symlinks.append(full_path)
    return symlinks

def resolve_symlink_target(symlink_path):
    """Get the target of a symlink, resolving relative paths."""
    try:
        target = os.readlink(symlink_path)
        if not os.path.isabs(target):
            # Convert relative path to absolute
            parent_dir = os.path.dirname(symlink_path)
            target = os.path.normpath(os.path.join(parent_dir, target))
        return target
    except Exception:
        return None

def detect_symlink_loops(root_path):
    """Detect all symlink loops using graph cycle detection."""
    symlinks = find_all_symlinks(root_path)
    
    # Build adjacency list
    graph = {}
    for symlink in symlinks:
        target = resolve_symlink_target(symlink)
        if target:
            graph[symlink] = target
    
    # Find all cycles using DFS
    loops = []
    visited = set()
    rec_stack = set()
    
    def dfs(node, path):
        if node in rec_stack:
            # Found a cycle
            cycle_start = path.index(node)
            cycle = path[cycle_start:]
            return [cycle]
        
        if node in visited:
            return []
        
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        cycles = []
        if node in graph:
            target = graph[node]
            # Only follow if target is also a symlink in our graph
            if target in graph or target in symlinks:
                cycles.extend(dfs(target, path[:]))
        
        path.pop()
        rec_stack.remove(node)
        return cycles
    
    # Check each symlink as potential cycle start
    for symlink in symlinks:
        if symlink not in visited:
            found_cycles = dfs(symlink, [])
            for cycle in found_cycles:
                # Normalize cycle to avoid duplicates
                if cycle and cycle not in loops:
                    # Check if this is a rotation of an existing loop
                    is_duplicate = False
                    for existing_loop in loops:
                        if len(cycle) == len(existing_loop):
                            if set(cycle) == set(existing_loop):
                                is_duplicate = True
                                break
                    if not is_duplicate:
                        loops.append(cycle)
    
    return loops, symlinks

def verify_no_loops(root_path):
    """Verify that no symlink loops exist."""
    symlinks = find_all_symlinks(root_path)
    
    for symlink in symlinks:
        visited = set()
        current = symlink
        
        for _ in range(len(symlinks) + 1):
            if current in visited:
                return False  # Loop detected
            visited.add(current)
            
            if not os.path.islink(current):
                break
            
            target = resolve_symlink_target(current)
            if not target or not os.path.exists(target):
                break
            
            current = target
    
    return True

def main():
    root_path = "/workspace/projects/"
    
    # Create the helper.py file first
    helper_dir = os.path.join(root_path, "shared/utils")
    os.makedirs(helper_dir, exist_ok=True)
    
    helper_content = '''"""
Shared utility module for common operations.
This module provides reusable utility functions across projects.
"""

from datetime import datetime
import json

def format_date(date_obj, format_str="%Y-%m-%d"):
    """Format a datetime object to string."""
    return date_obj.strftime(format_str)

def parse_json(json_string):
    """Parse JSON string and return dictionary."""
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        return None

def validate_input(value, min_length=1, max_length=100):
    """Validate input string length."""
    if not isinstance(value, str):
        return False
    return min_length <= len(value) <= max_length
'''
    
    helper_path = os.path.join(helper_dir, "helper.py")
    with open(helper_path, 'w') as f:
        f.write(helper_content)
    
    # Detect loops
    loops, all_symlinks = detect_symlink_loops(root_path)
    
    # Break loops by removing one symlink from each loop
    removed = []
    redirected = []
    
    for loop in loops:
        if loop and len(loop) > 0:
            # Remove the first symlink in the loop
            symlink_to_remove = loop[0]
            if os.path.exists(symlink_to_remove):
                os.unlink(symlink_to_remove)
                removed.append(symlink_to_remove)
    
    # Verify no loops remain
    verification_passed = verify_no_loops(root_path)
    
    # Create solution JSON
    solution = {
        "loops_found": len(loops),
        "symlinks_removed": removed,
        "symlinks_redirected": redirected,
        "verification_status": "PASS" if verification_passed else "FAIL"
    }
    
    # Save solution
    with open("/workspace/solution.json", 'w') as f:
        json.dump(solution, f, indent=2)
    
    print(f"Analysis complete. Found {len(loops)} loops.")
    print(f"Removed {len(removed)} symlinks.")
    print(f"Verification: {solution['verification_status']}")

if __name__ == "__main__":
    main()