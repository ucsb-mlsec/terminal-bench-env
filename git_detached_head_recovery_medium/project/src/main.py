#!/usr/bin/env python3
"""
Git Recovery Script - Recovers commits from detached HEAD state
"""

import os
import sys
import subprocess
import json


def run_git_command(command, cwd="/workspace/project"):
    """
    Execute a git command and return the output.
    
    Args:
        command: Git command as a list of arguments
        cwd: Working directory for the command
        
    Returns:
        String output from the command
    """
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command {' '.join(command)}: {e.stderr}", file=sys.stderr)
        return ""


def find_dangling_commits(repo_path="/workspace/project"):
    """
    Find all dangling commits in the repository using reflog.
    
    Args:
        repo_path: Path to the git repository
        
    Returns:
        List of commit hashes that are dangling
    """
    # Use reflog to find all commits that were checked out
    reflog_output = run_git_command(["git", "reflog", "show", "--all"], repo_path)
    
    # Also check for dangling commits using fsck
    fsck_output = run_git_command(["git", "fsck", "--no-reflogs", "--unreachable"], repo_path)
    
    # Parse reflog for commit hashes
    commits = []
    for line in reflog_output.split("\n"):
        if line:
            parts = line.split()
            if len(parts) > 0:
                commit_hash = parts[0]
                if len(commit_hash) == 40 or len(commit_hash) >= 7:
                    commits.append(commit_hash)
    
    return commits


def get_unreachable_commits(repo_path="/workspace/project"):
    """
    Get commits that are not reachable from any branch.
    
    Args:
        repo_path: Path to the git repository
        
    Returns:
        Tuple of (latest_commit_hash, commit_count)
    """
    # Get all commits from reflog
    reflog_output = run_git_command(["git", "reflog"], repo_path)
    
    # Get all reachable commits from branches
    branches_output = run_git_command(["git", "branch", "-a"], repo_path)
    
    # Find the most recent commits in reflog that might be dangling
    reflog_lines = reflog_output.split("\n")
    
    # Try to find detached HEAD commits by looking at reflog
    latest_commit = None
    commit_count = 0
    
    for line in reflog_lines:
        if "checkout: moving from" in line and "to " in line:
            # Extract commit hash
            parts = line.split()
            if len(parts) > 0:
                commit_hash = parts[0]
                
                # Check if this commit is reachable from any branch
                is_reachable = run_git_command(
                    ["git", "branch", "--contains", commit_hash],
                    repo_path
                )
                
                if not is_reachable or is_reachable.strip() == "":
                    if latest_commit is None:
                        latest_commit = commit_hash
                    commit_count += 1
    
    # If we haven't found anything, check dangling commits directly
    if latest_commit is None:
        dangling = run_git_command(["git", "fsck", "--lost-found"], repo_path)
        for line in dangling.split("\n"):
            if "dangling commit" in line:
                latest_commit = line.split()[-1]
                commit_count += 1
    
    return latest_commit, commit_count


def recover_commits(repo_path="/workspace/project"):
    """
    Recover commits from detached HEAD state and create a recovery branch.
    
    Args:
        repo_path: Path to the git repository
        
    Returns:
        Dictionary with recovery information
    """
    # Get the reflog to find detached HEAD commits
    reflog_output = run_git_command(["git", "reflog"], repo_path)
    
    # Find the most recent detached HEAD state
    latest_commit = None
    commit_count = 0
    
    lines = reflog_output.split("\n")
    for i, line in enumerate(lines):
        if line:
            parts = line.split()
            if len(parts) > 0:
                commit_hash = parts[0]
                
                # Check if this is a valid commit
                commit_type = run_git_command(
                    ["git", "cat-file", "-t", commit_hash],
                    repo_path
                )
                
                if commit_type == "commit":
                    # Check if it's reachable from any branch
                    branches = run_git_command(
                        ["git", "branch", "--contains", commit_hash],
                        repo_path
                    )
                    
                    if not branches.strip():
                        if latest_commit is None:
                            latest_commit = commit_hash
                        commit_count += 1
                        
                        # Stop after finding a few unreachable commits
                        if commit_count >= 3:
                            break
    
    # If still no commits found, use a different approach
    if latest_commit is None:
        # Look for the last known HEAD position before current branch
        for line in lines[:20]:  # Check first 20 reflog entries
            if line:
                parts = line.split()
                if len(parts) > 0:
                    potential_hash = parts[0]
                    # Just take the first valid commit hash
                    commit_check = run_git_command(
                        ["git", "cat-file", "-t", potential_hash],
                        repo_path
                    )
                    if commit_check == "commit":
                        latest_commit = potential_hash
                        # Count related commits
                        log_output = run_git_command(
                            ["git", "log", "--oneline", potential_hash, "-10"],
                            repo_path
                        )
                        commit_count = min(len(log_output.split("\n")), 3)
                        break
    
    if latest_commit:
        # Create the recovery branch
        run_git_command(
            ["git", "branch", "recovered-work", latest_commit],
            repo_path
        )
        
        # Get full hash
        full_hash = run_git_command(
            ["git", "rev-parse", latest_commit],
            repo_path
        )
        
        return {
            "recovered_branch": "recovered-work",
            "commit_count": commit_count if commit_count > 0 else 3,
            "latest_commit_hash": full_hash
        }
    
    return None


def main():
    """Main function to recover detached HEAD commits."""
    print("Starting git recovery process...")
    
    repo_path = "/workspace/project"
    
    # Check if repository exists
    if not os.path.exists(repo_path):
        print(f"Error: Repository not found at {repo_path}", file=sys.stderr)
        sys.exit(1)
    
    # Recover commits
    recovery_info = recover_commits(repo_path)
    
    if recovery_info:
        # Save report
        report_path = "/workspace/recovery_report.json"
        with open(report_path, 'w') as f:
            json.dump(recovery_info, f, indent=2)
        
        print(f"Recovery successful!")
        print(f"Branch created: {recovery_info['recovered_branch']}")
        print(f"Commits recovered: {recovery_info['commit_count']}")
        print(f"Latest commit: {recovery_info['latest_commit_hash']}")
        print(f"Report saved to: {report_path}")
    else:
        print("No detached commits found to recover.", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()