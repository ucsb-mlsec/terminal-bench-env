from flask import Flask, jsonify, request
import os
import psycopg2
from psycopg2 import pool
import redis
import logging
from datetime import datetime

# Initialize Flask application
app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from environment variables
# Database configuration
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_PORT = os.environ.get('DB_PORT', '5432')  # PostgreSQL default port
DB_NAME = os.environ.get('DB_NAME', 'apidb')
DB_USER = os.environ.get('DB_USER', 'admin')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'password')

# Redis configuration for caching
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = os.environ.get('REDIS_PORT', '6379')  # Redis default port

# Application configuration
DEBUG_MODE = os.environ.get('DEBUG', 'false').lower() == 'true'
API_KEY = os.environ.get('API_KEY', 'default-key')
MAX_CONNECTIONS = int(os.environ.get('MAX_DB_CONNECTIONS', '10'))

# Initialize database connection pool
try:
    logger.info(f"Attempting to connect to database at {DB_HOST}:{DB_PORT}")
    connection_pool = psycopg2.pool.SimpleConnectionPool(
        1,
        MAX_CONNECTIONS,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    if connection_pool:
        logger.info("Database connection pool created successfully")
except Exception as e:
    logger.error(f"Failed to create database connection pool: {str(e)}")
    connection_pool = None

# Initialize Redis connection
try:
    logger.info(f"Attempting to connect to Redis at {REDIS_HOST}:{REDIS_PORT}")
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=int(REDIS_PORT),
        db=0,
        decode_responses=True,
        socket_timeout=5
    )
    # Test the connection
    redis_client.ping()
    logger.info("Redis connection established successfully")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {str(e)}")
    redis_client = None


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify service status
    """
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'database': 'disconnected',
        'cache': 'disconnected'
    }
    
    # Check database connection
    if connection_pool:
        try:
            conn = connection_pool.getconn()
            cursor = conn.cursor()
            cursor.execute('SELECT 1')
            cursor.close()
            connection_pool.putconn(conn)
            health_status['database'] = 'connected'
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            health_status['status'] = 'degraded'
    
    # Check Redis connection
    if redis_client:
        try:
            redis_client.ping()
            health_status['cache'] = 'connected'
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
            health_status['status'] = 'degraded'
    
    status_code = 200 if health_status['status'] == 'healthy' else 503
    return jsonify(health_status), status_code


@app.route('/api/data', methods=['GET'])
def get_data():
    """
    Retrieve data from the database with Redis caching
    """
    # Verify API key
    provided_key = request.headers.get('X-API-Key')
    if provided_key != API_KEY:
        return jsonify({'error': 'Invalid API key'}), 401
    
    cache_key = 'api:data:all'
    
    # Try to get from cache first
    if redis_client:
        try:
            cached_data = redis_client.get(cache_key)
            if cached_data:
                logger.info("Returning cached data")
                return jsonify({'data': cached_data, 'source': 'cache'}), 200
        except Exception as e:
            logger.warning(f"Cache read failed: {str(e)}")
    
    # Get from database
    if not connection_pool:
        return jsonify({'error': 'Database unavailable'}), 503
    
    try:
        conn = connection_pool.getconn()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM api_data LIMIT 100')
        rows = cursor.fetchall()
        cursor.close()
        connection_pool.putconn(conn)
        
        data = [{'id': row[0], 'value': row[1]} for row in rows]
        
        # Cache the result
        if redis_client:
            try:
                redis_client.setex(cache_key, 300, str(data))
            except Exception as e:
                logger.warning(f"Cache write failed: {str(e)}")
        
        return jsonify({'data': data, 'source': 'database'}), 200
    except Exception as e:
        logger.error(f"Database query failed: {str(e)}")
        return jsonify({'error': 'Database query failed'}), 500


@app.route('/api/data', methods=['POST'])
def create_data():
    """
    Create new data entry in the database
    """
    # Verify API key
    provided_key = request.headers.get('X-API-Key')
    if provided_key != API_KEY:
        return jsonify({'error': 'Invalid API key'}), 401
    
    if not connection_pool:
        return jsonify({'error': 'Database unavailable'}), 503
    
    data = request.get_json()
    if not data or 'value' not in data:
        return jsonify({'error': 'Missing required field: value'}), 400
    
    try:
        conn = connection_pool.getconn()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO api_data (value) VALUES (%s) RETURNING id', (data['value'],))
        new_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        connection_pool.putconn(conn)
        
        return jsonify({'id': new_id, 'value': data['value']}), 201
    except Exception as e:
        logger.error(f"Database insert failed: {str(e)}")
        return jsonify({'error': 'Failed to create data'}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    Get application statistics
    """
    stats = {
        'app_name': 'api-service',
        'debug_mode': DEBUG_MODE,
        'max_connections': MAX_CONNECTIONS,
        'timestamp': datetime.now().isoformat()
    }
    return jsonify(stats), 200


@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """
    Clear the Redis cache
    """
    # Verify API key
    provided_key = request.headers.get('X-API-Key')
    if provided_key != API_KEY:
        return jsonify({'error': 'Invalid API key'}), 401
    
    if not redis_client:
        return jsonify({'error': 'Cache unavailable'}), 503
    
    try:
        redis_client.flushdb()
        return jsonify({'message': 'Cache cleared successfully'}), 200
    except Exception as e:
        logger.error(f"Cache clear failed: {str(e)}")
        return jsonify({'error': 'Failed to clear cache'}), 500


if __name__ == '__main__':
    logger.info(f"Starting Flask application on port 5000, debug={DEBUG_MODE}")
    app.run(host='0.0.0.0', port=5000, debug=DEBUG_MODE)