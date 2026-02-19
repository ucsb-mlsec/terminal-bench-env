#!/usr/bin/env python3

import sqlite3
import time
import sys

def main():
    db_path = '/workspace/blog_platform/blog.db'
    
    # Connect to database
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)
    
    # Define test search queries
    search_terms = [
        'performance',
        'optimization',
        'database',
        'python',
        'tutorial',
        'development',
        'programming',
        'technology',
        'software',
        'engineering'
    ]
    
    results = []
    all_passed = True
    
    print("Running search performance tests...\n")
    print(f"{'Search Term':<20} {'Time (ms)':<15} {'Status':<10}")
    print("-" * 50)
    
    for term in search_terms:
        search_pattern = f'%{term}%'
        
        # Measure query execution time
        start_time = time.perf_counter()
        cursor.execute("SELECT id, title FROM articles WHERE content LIKE ?", (search_pattern,))
        rows = cursor.fetchall()
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        
        passed = execution_time_ms < 100
        status = "PASS" if passed else "FAIL"
        
        if not passed:
            all_passed = False
        
        results.append({
            'term': term,
            'time_ms': execution_time_ms,
            'passed': passed,
            'count': len(rows)
        })
        
        print(f"{term:<20} {execution_time_ms:>10.2f} ms   {status:<10}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    total_queries = len(results)
    passed_queries = sum(1 for r in results if r['passed'])
    failed_queries = total_queries - passed_queries
    avg_time = sum(r['time_ms'] for r in results) / total_queries
    max_time = max(r['time_ms'] for r in results)
    
    print(f"Total queries:  {total_queries}")
    print(f"Passed:         {passed_queries}")
    print(f"Failed:         {failed_queries}")
    print(f"Average time:   {avg_time:.2f} ms")
    print(f"Max time:       {max_time:.2f} ms")
    print()
    
    if all_passed:
        print("✓ All queries completed in under 100ms")
        conn.close()
        sys.exit(0)
    else:
        print("✗ Some queries exceeded 100ms threshold")
        conn.close()
        sys.exit(1)

if __name__ == "__main__":
    main()