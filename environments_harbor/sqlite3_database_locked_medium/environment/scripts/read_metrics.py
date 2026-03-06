#!/usr/bin/env python3

import sqlite3
import sys

def read_metrics():
    try:
        # Connect to the database with default settings (no timeout)
        conn = sqlite3.connect('/workspace/data/metrics.db')
        cursor = conn.cursor()
        
        # Execute SELECT query to retrieve recent metrics
        cursor.execute("""
            SELECT timestamp, service_name, response_time_ms, status_code 
            FROM performance_metrics 
            ORDER BY timestamp DESC 
            LIMIT 10
        """)
        
        # Fetch and print results
        results = cursor.fetchall()
        print(f"Retrieved {len(results)} metrics:")
        for row in results:
            print(f"Timestamp: {row[0]}, Service: {row[1]}, Response Time: {row[2]}ms, Status: {row[3]}")
        
        # Also print some aggregated statistics
        cursor.execute("SELECT COUNT(*), AVG(response_time_ms) FROM performance_metrics")
        stats = cursor.fetchone()
        print(f"\nTotal metrics: {stats[0]}, Average response time: {stats[1]:.2f}ms")
        
        # Close the connection
        conn.close()
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    read_metrics()