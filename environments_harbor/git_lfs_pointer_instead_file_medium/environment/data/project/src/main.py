#!/usr/bin/env python3

import os
import sys
from pathlib import Path

def is_lfs_pointer(file_path):
    """
    Check if a file is a Git LFS pointer file.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        bool: True if the file is an LFS pointer, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline().strip()
            return first_line == "version https://git-lfs.github.com/spec/v1"
    except Exception:
        return False

def find_lfs_pointers(root_dir):
    """
    Recursively find all LFS pointer files in a directory.
    
    Args:
        root_dir: Root directory to search
        
    Returns:
        list: List of paths to LFS pointer files relative to root_dir
    """
    pointer_files = []
    root_path = Path(root_dir)
    
    for file_path in root_path.rglob('*'):
        if file_path.is_file():
            if is_lfs_pointer(file_path):
                relative_path = file_path.relative_to(root_path)
                pointer_files.append(relative_path)
    
    return pointer_files

def restore_from_backup(pointer_files, project_dir, backup_dir):
    """
    Restore LFS pointer files with actual content from backup.
    
    Args:
        pointer_files: List of relative paths to pointer files
        project_dir: Path to the project directory
        backup_dir: Path to the backup directory
        
    Returns:
        tuple: (restored_count, failed_count)
    """
    restored = 0
    failed = 0
    
    for relative_path in pointer_files:
        project_file = Path(project_dir) / relative_path
        backup_file = Path(backup_dir) / relative_path
        
        if backup_file.exists() and backup_file.is_file():
            try:
                # Copy backup file to project location
                with open(backup_file, 'rb') as src:
                    content = src.read()
                with open(project_file, 'wb') as dst:
                    dst.write(content)
                restored += 1
            except Exception as e:
                print(f"Error restoring {relative_path}: {e}", file=sys.stderr)
                failed += 1
        else:
            print(f"Backup not found for {relative_path}", file=sys.stderr)
            failed += 1
    
    return restored, failed

def main():
    """Main function to orchestrate LFS pointer restoration."""
    project_dir = "/workspace/project"
    backup_dir = "/workspace/backup"
    result_file = "/workspace/result.txt"
    
    # Find all LFS pointer files
    print("Scanning for LFS pointer files...")
    pointer_files = find_lfs_pointers(project_dir)
    found = len(pointer_files)
    
    print(f"Found {found} LFS pointer files")
    
    # Restore from backup
    if found > 0:
        print("Restoring files from backup...")
        restored, failed = restore_from_backup(pointer_files, project_dir, backup_dir)
    else:
        restored, failed = 0, 0
    
    # Write results
    with open(result_file, 'w') as f:
        f.write(f"FOUND={found}\n")
        f.write(f"RESTORED={restored}\n")
        f.write(f"FAILED={failed}\n")
    
    print(f"Results written to {result_file}")
    print(f"FOUND={found}, RESTORED={restored}, FAILED={failed}")

if __name__ == '__main__':
    main()