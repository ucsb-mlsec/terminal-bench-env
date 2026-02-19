#!/usr/bin/env python3

import os
import subprocess
import json

def run_command(cmd, cwd=None):
    """Run a shell command and return the output."""
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    return result.stdout.strip(), result.stderr.strip(), result.returncode

def main():
    parent_repo = "/workspace/project"
    submodule_path = "lib/shared"
    submodule_full_path = os.path.join(parent_repo, submodule_path)
    solution_file = "/workspace/solution.json"
    
    # Get the commit SHA that the parent repository expects for the submodule
    # This is stored in the parent's index
    cmd = f"git ls-files -s {submodule_path}"
    stdout, stderr, returncode = run_command(cmd, cwd=parent_repo)
    
    if returncode != 0 or not stdout:
        print(f"Error getting submodule info: {stderr}")
        return
    
    # Parse the ls-files output: mode SHA stage path
    parts = stdout.split()
    expected_commit = parts[1]
    
    print(f"Parent repository expects submodule at commit: {expected_commit}")
    
    # Get the current commit in the submodule's working directory
    cmd = "git rev-parse HEAD"
    current_commit, stderr, returncode = run_command(cmd, cwd=submodule_full_path)
    
    if returncode != 0:
        print(f"Error getting current submodule commit: {stderr}")
        return
    
    print(f"Submodule is currently at commit: {current_commit}")
    
    # Check if there's a mismatch
    was_mismatched = (current_commit != expected_commit)
    
    if was_mismatched:
        print(f"Mismatch detected! Checking out expected commit...")
        
        # Checkout the expected commit in the submodule
        cmd = f"git checkout {expected_commit}"
        stdout, stderr, returncode = run_command(cmd, cwd=submodule_full_path)
        
        if returncode != 0:
            print(f"Error checking out commit: {stderr}")
            return
        
        print(f"Successfully checked out commit {expected_commit}")
        
        # Verify the checkout
        current_commit, stderr, returncode = run_command("git rev-parse HEAD", cwd=submodule_full_path)
        if current_commit == expected_commit:
            print("Verification successful: Submodule is now at the expected commit")
        else:
            print(f"Verification failed: Expected {expected_commit}, got {current_commit}")
    else:
        print("No mismatch detected. Submodule is already at the expected commit.")
    
    # Verify clean state
    cmd = "git status --porcelain"
    status_output, stderr, returncode = run_command(cmd, cwd=parent_repo)
    
    if status_output:
        print(f"Repository status:\n{status_output}")
    else:
        print("Repository is in a clean state")
    
    # Save the solution
    solution = {
        "expected_commit": expected_commit,
        "was_mismatched": was_mismatched
    }
    
    with open(solution_file, 'w') as f:
        json.dump(solution, f, indent=2)
    
    print(f"\nSolution saved to {solution_file}")
    print(json.dumps(solution, indent=2))

if __name__ == "__main__":
    main()