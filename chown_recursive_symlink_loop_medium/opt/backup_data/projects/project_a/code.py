#!/usr/bin/env python3

import os
import json
from pathlib import Path

def find_symlink_loops(root_dir):
    """
    Scan directory structure and identify circular symbolic link chains.
    """
    loops = []
    visited_chains = set()
    
    def resolve_symlink_chain(start_path, chain=None, visited_targets=None):
        """
        Follow a symlink chain and detect if it loops back.
        Returns a loop object if a circular reference is found, None otherwise.
        """
        if chain is None:
            chain = []
        if visited_targets is None:
            visited_targets = set()
        
        # Get the target of the current symlink
        try:
            target = os.readlink(start_path)
            # Convert relative paths to absolute
            if not os.path.isabs(target):
                target = os.path.normpath(os.path.join(os.path.dirname(start_path), target))
            else:
                target = os.path.normpath(target)
        except (OSError, FileNotFoundError):
            return None
        
        # Add current symlink to chain
        chain.append(start_path)
        
        # Check if target resolves to a directory we've already visited in this chain
        for i, visited_link in enumerate(chain[:-1]):
            try:
                visited_target = os.readlink(visited_link)
                if not os.path.isabs(visited_target):
                    visited_target = os.path.normpath(os.path.join(os.path.dirname(visited_link), visited_target))
                else:
                    visited_target = os.path.normpath(visited_target)
                
                # Check if current target points back to a previous target in chain
                if target == visited_target:
                    return {
                        "chain": chain.copy(),
                        "creates_loop": start_path
                    }
                    
                # Check if current target points to parent of a previous link in chain
                if target == os.path.dirname(visited_link):
                    return {
                        "chain": chain.copy(),
                        "creates_loop": start_path
                    }
            except:
                pass
        
        # Check if we've seen this target in our chain
        if target in visited_targets:
            return {
                "chain": chain.copy(),
                "creates_loop": start_path
            }
        
        visited_targets.add(target)
        
        # If target exists and is a directory with symlinks, continue following
        if os.path.isdir(target) and not os.path.islink(target):
            try:
                for entry in os.listdir(target):
                    entry_path = os.path.join(target, entry)
                    if os.path.islink(entry_path):
                        result = resolve_symlink_chain(entry_path, chain.copy(), visited_targets.copy())
                        if result:
                            return result
            except (PermissionError, OSError):
                pass
        elif os.path.islink(target):
            # Continue following the chain
            result = resolve_symlink_chain(target, chain.copy(), visited_targets.copy())
            if result:
                return result
        
        return None
    
    def scan_directory(directory):
        """
        Recursively scan directory for symlinks and check each for loops.
        """
        try:
            entries = os.listdir(directory)
        except (PermissionError, OSError):
            return
        
        for entry in entries:
            entry_path = os.path.join(directory, entry)
            
            if os.path.islink(entry_path):
                # Check if this symlink creates a loop
                loop = resolve_symlink_chain(entry_path)
                
                if loop:
                    # Create a normalized representation to avoid duplicates
                    chain_normalized = tuple(sorted(loop["chain"]))
                    if chain_normalized not in visited_chains:
                        visited_chains.add(chain_normalized)
                        loops.append(loop)
            
            # Recurse into directories (but not through symlinks to avoid loops)
            if os.path.isdir(entry_path) and not os.path.islink(entry_path):
                scan_directory(entry_path)
    
    # Start scanning from root
    scan_directory(root_dir)
    
    return loops

def main():
    backup_dir = "/opt/backup_data"
    output_file = "/tmp/symlink_loops.json"
    
    # Find all circular symlink chains
    loops = find_symlink_loops(backup_dir)
    
    # Prepare output
    result = {
        "total_loops": len(loops),
        "loops": loops
    }
    
    # Write to output file
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Analysis complete. Found {len(loops)} circular symlink chain(s).")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()