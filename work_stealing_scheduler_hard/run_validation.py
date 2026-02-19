#!/usr/bin/env python3

import os
import sys
import json
import glob
from simulator import Simulator
from scheduler import Scheduler


def load_workload(filepath):
    """Reads a JSON workload file and returns the parsed workload configuration."""
    try:
        with open(filepath, 'r') as f:
            workload = json.load(f)
        return workload
    except FileNotFoundError:
        print(f"Error: Workload file not found: {filepath}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in workload file: {filepath}")
        sys.exit(1)


def run_test(workload_file):
    """
    Loads the workload, creates scheduler and simulator instances,
    runs the simulation, and returns the metrics dict.
    """
    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(workload_file)}")
    print(f"{'='*60}")
    
    workload = load_workload(workload_file)
    
    scheduler = Scheduler()
    simulator = Simulator(scheduler, workload)
    
    try:
        metrics = simulator.run()
        
        print(f"  Tasks Completed: {metrics['tasks_completed']}")
        print(f"  Idle Violations: {metrics['idle_violations']}")
        print(f"  Tasks Stolen: {metrics['tasks_stolen']}")
        print(f"  Status: {'✓ PASS' if metrics['idle_violations'] == 0 else '✗ FAIL'}")
        
        return metrics
    except Exception as e:
        print(f"  Error during simulation: {e}")
        return {
            'tasks_completed': 0,
            'idle_violations': float('inf'),
            'tasks_stolen': 0,
            'passed': False
        }


def write_validation_result(output_file, results):
    """
    Writes the aggregate results to output_file in the required format.
    """
    all_tests_passed = "yes" if results['all_passed'] else "no"
    
    with open(output_file, 'w') as f:
        f.write(f"TOTAL_TASKS_COMPLETED={results['total_tasks_completed']}\n")
        f.write(f"IDLE_VIOLATIONS={results['total_idle_violations']}\n")
        f.write(f"TASKS_STOLEN={results['total_tasks_stolen']}\n")
        f.write(f"ALL_TESTS_PASSED={all_tests_passed}\n")


def main():
    """
    Main function that discovers workload files, runs tests,
    aggregates metrics, and returns aggregate results.
    """
    workload_dir = '/opt/task_processor/workloads/'
    workload_files = sorted(glob.glob(os.path.join(workload_dir, '*.json')))
    
    if not workload_files:
        print(f"Error: No workload files found in {workload_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(workload_files)} workload(s) to test")
    
    total_tasks_completed = 0
    total_idle_violations = 0
    total_tasks_stolen = 0
    all_passed = True
    
    for workload_file in workload_files:
        metrics = run_test(workload_file)
        
        total_tasks_completed += metrics.get('tasks_completed', 0)
        total_idle_violations += metrics.get('idle_violations', 0)
        total_tasks_stolen += metrics.get('tasks_stolen', 0)
        
        if metrics.get('idle_violations', 0) > 0:
            all_passed = False
    
    # Check if tasks were actually stolen (must be > 0)
    if total_tasks_stolen == 0:
        all_passed = False
    
    results = {
        'total_tasks_completed': total_tasks_completed,
        'total_idle_violations': total_idle_violations,
        'total_tasks_stolen': total_tasks_stolen,
        'all_passed': all_passed
    }
    
    return results


if __name__ == '__main__':
    print("\n" + "="*60)
    print("TASK PROCESSOR VALIDATION SUITE")
    print("="*60)
    
    results = main()
    
    print("\n" + "="*60)
    print("AGGREGATE RESULTS")
    print("="*60)
    print(f"Total Tasks Completed: {results['total_tasks_completed']}")
    print(f"Total Idle Violations: {results['total_idle_violations']}")
    print(f"Total Tasks Stolen: {results['total_tasks_stolen']}")
    print(f"All Tests Passed: {'YES ✓' if results['all_passed'] else 'NO ✗'}")
    print("="*60)
    
    output_file = '/opt/task_processor/validation_result.txt'
    write_validation_result(output_file, results)
    
    print(f"\nValidation results written to: {output_file}")
    
    if results['all_passed']:
        print("\n🎉 SUCCESS: All validations passed!")
        sys.exit(0)
    else:
        print("\n❌ FAILURE: Some validations failed.")
        sys.exit(1)