from flask import Blueprint, jsonify
import time
import psutil
import os

metrics_blueprint = Blueprint('metrics', __name__)

# Module-level variables for tracking
start_time = time.time()
request_counter = 0
health_check_counter = 0

@metrics_blueprint.route('/metrics/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify service availability"""
    global health_check_counter
    health_check_counter += 1
    
    return jsonify({
        'status': 'healthy',
        'service': 'webapp',
        'checks_performed': health_check_counter,
        'timestamp': time.time()
    })

@metrics_blueprint.route('/metrics/stats', methods=['GET'])
def get_stats():
    """Return application statistics"""
    global request_counter
    request_counter += 1
    
    uptime = time.time() - start_time
    
    return jsonify({
        'uptime_seconds': uptime,
        'total_requests': request_counter,
        'health_checks': health_check_counter,
        'pid': os.getpid()
    })

@metrics_blueprint.route('/metrics/performance', methods=['GET'])
def get_performance():
    """Return performance metrics including memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return jsonify({
        'memory_rss_mb': memory_info.rss / 1024 / 1024,
        'memory_vms_mb': memory_info.vms / 1024 / 1024,
        'cpu_percent': process.cpu_percent(interval=0.1),
        'num_threads': process.num_threads(),
        'uptime_seconds': time.time() - start_time
    })