#!/usr/bin/env python3

import sqlite3
import time
from datetime import datetime

def calculate_hourly_summaries(conn):
    """Calculate hourly statistics from sensor readings."""
    try:
        cursor = conn.cursor()
        
        # Get the latest processed hour
        query = """
        SELECT 
            (recorded_at / 3600) * 3600 as hour_timestamp,
            AVG(temperature) as avg_temp,
            AVG(humidity) as avg_humidity,
            COUNT(*) as reading_count
        FROM readings
        GROUP BY hour_timestamp
        ORDER BY hour_timestamp DESC
        LIMIT 10
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        print(f"\n=== Hourly Summary Report ({datetime.now()}) ===")
        for row in results:
            hour_ts, avg_temp, avg_humidity, count = row
            hour_str = datetime.fromtimestamp(hour_ts).strftime('%Y-%m-%d %H:00')
            print(f"Hour: {hour_str} | Avg Temp: {avg_temp:.2f}°C | Avg Humidity: {avg_humidity:.2f}% | Readings: {count}")
        
        cursor.close()
        return True
        
    except sqlite3.Error as e:
        print(f"Database error during summary calculation: {e}")
        return False

def get_total_readings(conn):
    """Get total number of readings in database."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM readings")
        count = cursor.fetchone()[0]
        cursor.close()
        return count
    except sqlite3.Error as e:
        print(f"Error getting reading count: {e}")
        return 0

def main():
    """Main processing loop for summary generation."""
    print("Starting summary processor...")
    
    # Connect to database with default settings (problem area)
    conn = sqlite3.connect('data/sensors.db')
    
    try:
        print(f"Connected to database. Total readings: {get_total_readings(conn)}")
        
        # Process summaries in a loop
        iteration = 0
        while True:
            iteration += 1
            print(f"\n--- Processing iteration {iteration} ---")
            
            success = calculate_hourly_summaries(conn)
            
            if success:
                print("Summary calculation completed successfully")
            else:
                print("Summary calculation failed")
            
            # Wait before next processing cycle
            print("Waiting 10 seconds before next summary...")
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\nShutting down summary processor...")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        conn.close()
        print("Database connection closed")

if __name__ == "__main__":
    main()