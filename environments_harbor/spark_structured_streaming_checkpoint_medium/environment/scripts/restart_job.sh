#!/usr/bin/env python3

import os
import re
import sys

# Configuration
APP_DIR = "/opt/streaming_app"
OLD_PATH = "/mnt/old_storage/checkpoints"
NEW_PATH = "/mnt/new_storage/streaming_checkpoints"
SUMMARY_FILE = "/opt/streaming_app/migration_summary.txt"

def is_text_file(filepath):
    """Check if a file is likely a text file."""
    try:
        with open(filepath, 'rb') as f:
            chunk = f.read(1024)
            if b'\0' in chunk:
                return False
        return True
    except:
        return False

def should_process_file(filepath, filename):
    """Determine if a file should be processed."""
    # Skip hidden files
    if filename.startswith('.'):
        return False
    
    # Skip the summary file itself
    if filepath == SUMMARY_FILE:
        return False
    
    # Check if it's a text file
    if not os.path.isfile(filepath):
        return False
    
    return is_text_file(filepath)

def update_checkpoint_paths(filepath):
    """Update checkpoint paths in a file and return the number of replacements."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except:
        return 0
    
    # Check if the old path exists in the file
    if OLD_PATH not in content:
        return 0
    
    # Replace all occurrences
    updated_content = content.replace(OLD_PATH, NEW_PATH)
    
    # Count replacements
    replacements = content.count(OLD_PATH)
    
    if replacements > 0:
        # Write the updated content back
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(updated_content)
        except:
            return 0
    
    return replacements

def main():
    """Main function to perform the migration."""
    
    # Ensure the app directory exists
    if not os.path.exists(APP_DIR):
        print(f"Error: Application directory {APP_DIR} does not exist.")
        sys.exit(1)
    
    files_modified = 0
    total_replacements = 0
    
    # Walk through the directory tree
    for root, dirs, files in os.walk(APP_DIR):
        # Remove hidden directories from the walk
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for filename in files:
            filepath = os.path.join(root, filename)
            
            if should_process_file(filepath, filename):
                replacements = update_checkpoint_paths(filepath)
                if replacements > 0:
                    files_modified += 1
                    total_replacements += replacements
                    print(f"Updated {filepath}: {replacements} replacement(s)")
    
    # Write the migration summary
    try:
        os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)
        with open(SUMMARY_FILE, 'w') as f:
            f.write(f"files_modified={files_modified}\n")
            f.write(f"total_replacements={total_replacements}\n")
            f.write(f"old_path={OLD_PATH}\n")
            f.write(f"new_path={NEW_PATH}\n")
        print(f"\nMigration summary written to {SUMMARY_FILE}")
    except Exception as e:
        print(f"Error writing summary file: {e}")
        sys.exit(1)
    
    print(f"\nMigration completed:")
    print(f"Files modified: {files_modified}")
    print(f"Total replacements: {total_replacements}")

if __name__ == "__main__":
    main()