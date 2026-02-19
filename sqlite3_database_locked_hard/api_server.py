#!/usr/bin/env python3

from flask import Flask, request, jsonify
import sqlite3
import time

app = Flask(__name__)

def get_db_connection():
    """Create and return a database connection"""
    conn = sqlite3.connect('data/sensors.db')
    conn.row_factory = sqlite3.Row
    return conn

def insert_reading(sensor_id, temperature, humidity):
    """Insert a sensor reading into the database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        recorded_at = int(time.time())
        
        cursor.execute('''
            INSERT INTO readings (sensor_id, temperature, humidity, recorded_at)
            VALUES (?, ?, ?, ?)
        ''', (sensor_id, temperature, humidity, recorded_at))
        
        conn.commit()
        reading_id = cursor.lastrowid
        conn.close()
        
        return reading_id
    except Exception as e:
        if conn:
            conn.close()
        raise e

@app.route('/reading', methods=['POST'])
def add_reading():
    """Endpoint to receive sensor readings"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        sensor_id = data.get('sensor_id')
        temperature = data.get('temperature')
        humidity = data.get('humidity')
        
        if sensor_id is None or temperature is None or humidity is None:
            return jsonify({'error': 'Missing required fields'}), 400
        
        reading_id = insert_reading(sensor_id, temperature, humidity)
        
        return jsonify({
            'status': 'success',
            'reading_id': reading_id,
            'sensor_id': sensor_id,
            'temperature': temperature,
            'humidity': humidity
        }), 201
        
    except sqlite3.Error as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'running'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)