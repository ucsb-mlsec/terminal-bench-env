#!/usr/bin/env python3

import os
import time
import csv
import logging
from io import StringIO
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/webapp/app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


def validate_row(row_data, row_number):
    """
    Validate a single CSV row - simulates validation logic
    This takes approximately 10ms per row
    """
    time.sleep(0.01)  # Simulate validation processing
    
    # Basic validation checks
    if not row_data:
        raise ValueError(f"Empty row at line {row_number}")
    
    return True


def transform_row(row_data):
    """
    Transform CSV row data - simulates data transformation
    This takes approximately 15ms per row
    """
    time.sleep(0.015)  # Simulate transformation operations
    
    # Simulate data transformation (normalization, type conversion, etc.)
    transformed = {}
    for key, value in row_data.items():
        transformed[key.strip().lower()] = value.strip()
    
    return transformed


def insert_to_database(row_data):
    """
    Simulate database insertion operation
    This takes approximately 20ms per row due to DB I/O
    """
    time.sleep(0.02)  # Simulate database insertion
    
    # In a real application, this would be actual DB operations like:
    # db.session.add(DataModel(**row_data))
    # db.session.commit()
    
    return True


def process_csv_file(file_content):
    """
    Process CSV file with validation, transformation, and database insertion
    
    Processing breakdown per 1000 rows:
    - Validation: ~10 seconds (10ms per row)
    - Transformation: ~15 seconds (15ms per row)
    - Database insertion: ~20 seconds (20ms per row)
    - Overhead and I/O: ~10 seconds
    Total: ~55 seconds per 1000 rows
    
    For 2000 rows: ~110 seconds
    """
    csv_reader = csv.DictReader(StringIO(file_content))
    
    processed_rows = 0
    errors = []
    
    for row_number, row in enumerate(csv_reader, start=1):
        try:
            # Step 1: Validation
            validate_row(row, row_number)
            
            # Step 2: Transformation
            transformed_data = transform_row(row)
            
            # Step 3: Database insertion
            insert_to_database(transformed_data)
            
            processed_rows += 1
            
            # Log progress every 100 rows
            if processed_rows % 100 == 0:
                logger.info(f"Processed {processed_rows} rows...")
                
        except Exception as e:
            error_msg = f"Error processing row {row_number}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
    
    return processed_rows, errors


@app.route('/upload', methods=['POST'])
def upload_csv():
    """
    Handle CSV file upload and processing
    Accepts CSV files and processes them with validation, transformation, and DB insertion
    """
    start_time = time.time()
    
    logger.info("Received CSV upload request")
    
    # Check if file is present in request
    if 'file' not in request.files:
        logger.error("No file provided in request")
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        logger.error("Empty filename")
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.csv'):
        logger.error(f"Invalid file type: {file.filename}")
        return jsonify({'error': 'Only CSV files are allowed'}), 400
    
    try:
        # Read file content
        file_content = file.read().decode('utf-8')
        logger.info(f"File '{file.filename}' read successfully, size: {len(file_content)} bytes")
        
        # Count rows for logging
        row_count = len(file_content.split('\n')) - 1  # Subtract header
        logger.info(f"Starting processing of {row_count} rows from '{file.filename}'")
        
        # Process the CSV file
        processed_rows, errors = process_csv_file(file_content)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        logger.info(f"Completed processing {processed_rows} rows in {processing_time:.2f} seconds")
        
        if errors:
            logger.warning(f"Processing completed with {len(errors)} errors")
            return jsonify({
                'status': 'partial_success',
                'processed_rows': processed_rows,
                'errors': errors[:10],  # Return first 10 errors
                'error_count': len(errors),
                'processing_time': round(processing_time, 2)
            }), 207
        
        return jsonify({
            'status': 'success',
            'processed_rows': processed_rows,
            'processing_time': round(processing_time, 2)
        }), 200
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Fatal error during file processing after {processing_time:.2f} seconds: {str(e)}")
        return jsonify({
            'error': 'File processing failed',
            'details': str(e),
            'processing_time': round(processing_time, 2)
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'healthy'}), 200


@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API information"""
    return jsonify({
        'message': 'CSV Processing API',
        'endpoints': {
            '/upload': 'POST - Upload CSV file for processing',
            '/health': 'GET - Health check'
        }
    }), 200


if __name__ == '__main__':
    # For development only - use Gunicorn in production
    app.run(host='0.0.0.0', port=5000, debug=False)