#!/usr/bin/env python3

import sqlite3
import time

def query_by_date(db_path, date_str):
    """
    Legacy query function that demonstrates the slow query pattern.
    This approach uses LIKE on the timestamp column without proper indexing,
    resulting in full table scans and poor performance.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    start_time = time.time()
    
    # This LIKE query requires scanning the entire table
    # because timestamp is stored as TEXT without proper indexing
    query = "SELECT * FROM readings WHERE timestamp LIKE ?"
    cursor.execute(query, (f"{date_str}%",))
    
    results = cursor.fetchall()
    
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    conn.close()
    
    return results, execution_time

if __name__ == "__main__":
    db_path = './sensor_data.db'
    
    # Sample query for a specific date
    test_date = '2022-06-15'
    
    print(f"Executing legacy query for date: {test_date}")
    print("This demonstrates the slow query pattern...")
    
    results, exec_time = query_by_date(db_path, test_date)
    
    print(f"Number of results: {len(results)}")
    print(f"Execution time: {exec_time:.2f} ms")
    
    if exec_time > 500:
        print(f"WARNING: Query took {exec_time:.2f}ms, exceeding 500ms threshold!")
    
    # This is the legacy slow approach that requires optimization
    # The LIKE operator on text timestamps without proper indexing
    # causes SQLite to perform full table scans on large datasets
