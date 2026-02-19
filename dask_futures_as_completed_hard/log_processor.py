#!/usr/bin/env python3

import concurrent.futures
import time
import random
import json
from dataclasses import dataclass, asdict
from typing import List, Dict

@dataclass
class LogEntry:
    id: int
    source: str
    timestamp: float
    message: str
    severity: str
    processed_at: float = None

class ValidationError(Exception):
    pass

def validate_log_entry(entry: LogEntry):
    """Validates a log entry. Randomly fails for ~5% of entries."""
    # Simulate validation that fails ~5% of the time
    if random.random() < 0.05:
        raise ValidationError(f"Validation failed for log entry {entry.id}")
    
    # Basic validation checks
    if not entry.message:
        raise ValidationError(f"Empty message in log entry {entry.id}")
    if entry.severity not in ['INFO', 'WARNING', 'ERROR', 'DEBUG', 'CRITICAL']:
        raise ValidationError(f"Invalid severity in log entry {entry.id}")

def process_log_entry(entry: LogEntry, processing_time: float) -> LogEntry:
    """
    Processes a single log entry.
    Simulates work by sleeping, then validates and enriches the entry.
    """
    # Simulate processing work
    time.sleep(processing_time)
    
    # Validate the entry (may raise ValidationError)
    validate_log_entry(entry)
    
    # Enrich the entry with processing timestamp
    entry.processed_at = time.time()
    
    return entry

def process_logs(log_entries: List[tuple], max_workers: int = 10) -> Dict:
    """
    CURRENT BROKEN IMPLEMENTATION
    
    This function processes logs but has severe performance issues.
    It waits for ALL futures to complete before processing any results,
    creating a bottleneck where fast-completing tasks wait for slow ones.
    """
    print("Starting log processing (BROKEN implementation)...")
    start_time = time.time()
    
    processed = []
    failed = []
    
    # Create executor
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    
    # Submit all tasks and collect futures
    # PROBLEM: Collecting all futures in a list
    all_futures = []
    for entry, processing_time in log_entries:
        future = executor.submit(process_log_entry, entry, processing_time)
        all_futures.append(future)
    
    # MAJOR BOTTLENECK: Wait for ALL futures to complete before processing ANY results
    # This defeats the purpose of async processing - we can't process entries
    # as they complete, we have to wait for the slowest one
    print("Waiting for ALL tasks to complete before processing results...")
    done, pending = concurrent.futures.wait(
        all_futures, 
        timeout=None,  # Wait indefinitely
        return_when=concurrent.futures.ALL_COMPLETED  # Wait for everything!
    )
    
    # Only after everything is done, process results
    print(f"All tasks complete. Processing {len(done)} results...")
    for future in done:
        try:
            result = future.result()
            processed.append(result)
        except ValidationError as e:
            failed.append(str(e))
        except Exception as e:
            failed.append(f"Unexpected error: {str(e)}")
    
    executor.shutdown()
    
    processing_time = time.time() - start_time
    
    print(f"Processing complete:")
    print(f"  Total processed: {len(processed)}")
    print(f"  Total failed: {len(failed)}")
    print(f"  Processing time: {processing_time:.2f} seconds")
    
    return {
        'total_processed': len(processed),
        'total_failed': len(failed),
        'processing_time_seconds': processing_time,
        'processed_entries': processed
    }

def generate_sample_logs(num_entries: int = 100, num_sources: int = 5) -> List[tuple]:
    """
    Generates sample log entries with varying processing times.
    Returns list of tuples: (LogEntry, processing_time)
    """
    sources = [f"source_{i}" for i in range(num_sources)]
    severities = ['INFO', 'WARNING', 'ERROR', 'DEBUG', 'CRITICAL']
    
    log_entries = []
    
    for i in range(num_entries):
        source = random.choice(sources)
        # Different sources have different processing speeds
        source_num = int(source.split('_')[1])
        base_time = 0.1 + (source_num * 0.5)  # 0.1 to 2.6 seconds
        processing_time = base_time + random.uniform(-0.05, 0.2)
        
        entry = LogEntry(
            id=i,
            source=source,
            timestamp=time.time(),
            message=f"Log message {i} from {source}",
            severity=random.choice(severities)
        )
        
        log_entries.append((entry, processing_time))
    
    return log_entries

if __name__ == "__main__":
    # Generate sample logs
    print("Generating sample log entries...")
    logs = generate_sample_logs(num_entries=1000, num_sources=5)
    
    # Process logs using broken implementation
    results = process_logs(logs, max_workers=10)
    
    # The broken implementation doesn't calculate completion order deviation
    # because it waits for everything at once
    results['average_completion_order_deviation'] = 0.0
    
    # Save results
    output = {
        'total_processed': results['total_processed'],
        'total_failed': results['total_failed'],
        'processing_time_seconds': results['processing_time_seconds'],
        'average_completion_order_deviation': results['average_completion_order_deviation']
    }
    
    with open('/workspace/results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to /workspace/results.json")
    print(f"Note: This implementation will likely TIMEOUT (>60s) due to inefficient processing!")