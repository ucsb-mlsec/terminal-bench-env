#!/usr/bin/env python3

from flask import Flask, jsonify
import psycopg2
import time
import random
import os
from psycopg2 import pool

app = Flask(__name__)

# Database connection pool - limited size can cause contention
# This small pool size (2 connections) causes issues under high load
DB_CONNECTION_POOL = psycopg2.pool.SimpleConnectionPool(
    1, 2,
    user=os.getenv('DB_USER', 'appuser'),
    password=os.getenv('DB_PASSWORD', 'password'),
    host=os.getenv('DB_HOST', 'localhost'),
    port=os.getenv('DB_PORT', '5432'),
    database=os.getenv('DB_NAME', 'appdb')
)

@app.route('/health')
def health_check():
    """
    Health check endpoint that performs multiple checks.
    
    PERFORMANCE ISSUE: This endpoint has several problems:
    1. Database connectivity check can take 2-5 seconds when connection pool is exhausted
    2. No timeout configured for database operations
    3. Synchronous blocking calls that can queue up under high load
    4. The health check competes with application traffic for limited connection pool
    """
    
    health_status = {
        "status": "healthy",
        "checks": {}
    }
    
    start_time = time.time()
    
    # Memory check - this is fast
    try:
        import psutil
        memory = psutil.virtual_memory()
        if memory.percent < 90:
            health_status["checks"]["memory"] = "ok"
        else:
            health_status["checks"]["memory"] = "warning"
            health_status["status"] = "degraded"
    except ImportError:
        health_status["checks"]["memory"] = "ok"
    
    # Database connectivity check - THIS IS THE PROBLEM
    # When the application is under high load (1000 req/s), the small connection
    # pool gets exhausted. Health checks then wait for available connections.
    # This can take 2-5 seconds, causing Route53 health check timeouts.
    try:
        # Attempt to get connection from pool - this blocks if pool is exhausted
        conn = DB_CONNECTION_POOL.getconn()
        
        if conn:
            cursor = conn.cursor()
            
            # Execute a simple query to verify database is responsive
            # In reality, even this simple query can be slow if database is under load
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            
            # Simulate occasional slow query execution (3-8% of requests)
            # This represents real-world scenario where database can be slow
            if random.random() < 0.05:
                time.sleep(random.uniform(2.0, 4.5))
            
            cursor.close()
            DB_CONNECTION_POOL.putconn(conn)
            
            health_status["checks"]["database"] = "ok"
        else:
            health_status["checks"]["database"] = "unavailable"
            health_status["status"] = "unhealthy"
            
    except Exception as e:
        health_status["checks"]["database"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Add timing information
    elapsed_time = time.time() - start_time
    health_status["response_time_ms"] = round(elapsed_time * 1000, 2)
    
    # Return appropriate status code
    if health_status["status"] == "healthy":
        return jsonify(health_status), 200
    elif health_status["status"] == "degraded":
        return jsonify(health_status), 200
    else:
        return jsonify(health_status), 503

@app.route('/')
def index():
    return jsonify({"message": "Application is running"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)