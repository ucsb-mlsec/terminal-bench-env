#!/usr/bin/env python3

import os
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask application
app = Flask(__name__)

# File size limits (in bytes)
# Avatar uploads: Maximum 5MB for user profile pictures
MAX_AVATAR_SIZE = 5 * 1024 * 1024  # 5MB

# Document uploads: Maximum 50MB for PDF, Word docs, spreadsheets, etc.
MAX_DOCUMENT_SIZE = 50 * 1024 * 1024  # 50MB

# Media uploads: Maximum 200MB for videos, high-res images, audio files
MAX_MEDIA_SIZE = 200 * 1024 * 1024  # 200MB

# Upload directory configuration
UPLOAD_FOLDER = '/tmp/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_AVATAR_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_DOCUMENT_EXTENSIONS = {'pdf', 'doc', 'docx', 'xls', 'xlsx', 'txt', 'csv'}
ALLOWED_MEDIA_EXTENSIONS = {'mp4', 'avi', 'mov', 'mp3', 'wav', 'png', 'jpg', 'jpeg'}


def allowed_file(filename, allowed_extensions):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def get_file_size(file):
    """Get the size of an uploaded file"""
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    return size


@app.route('/api/upload/avatar', methods=['POST'])
def upload_avatar():
    """
    Handle avatar image uploads
    Maximum size: 5MB
    Allowed formats: PNG, JPG, JPEG, GIF
    """
    try:
        logger.info("Avatar upload request received")
        
        if 'file' not in request.files:
            logger.warning("No file part in avatar upload request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            logger.warning("Empty filename in avatar upload request")
            return jsonify({'error': 'No file selected'}), 400
        
        # Get file size
        file_size = get_file_size(file)
        logger.info(f"Avatar upload attempt - filename: {file.filename}, size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
        
        # Check file size limit
        if file_size > MAX_AVATAR_SIZE:
            logger.error(f"Avatar upload rejected - file size {file_size} exceeds limit of {MAX_AVATAR_SIZE} bytes")
            return jsonify({
                'error': 'File too large',
                'max_size': f'{MAX_AVATAR_SIZE / (1024*1024):.0f}MB',
                'uploaded_size': f'{file_size / (1024*1024):.2f}MB'
            }), 413
        
        # Check file extension
        if not allowed_file(file.filename, ALLOWED_AVATAR_EXTENSIONS):
            logger.warning(f"Avatar upload rejected - invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type', 'allowed': list(ALLOWED_AVATAR_EXTENSIONS)}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, 'avatar_' + filename)
        file.save(filepath)
        
        logger.info(f"Avatar uploaded successfully - {filename} ({file_size} bytes)")
        return jsonify({
            'success': True,
            'message': 'Avatar uploaded successfully',
            'filename': filename,
            'size': file_size
        }), 200
        
    except Exception as e:
        logger.exception(f"Error processing avatar upload: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/upload/document', methods=['POST'])
def upload_document():
    """
    Handle document uploads
    Maximum size: 50MB
    Allowed formats: PDF, DOC, DOCX, XLS, XLSX, TXT, CSV
    """
    try:
        logger.info("Document upload request received")
        
        if 'file' not in request.files:
            logger.warning("No file part in document upload request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            logger.warning("Empty filename in document upload request")
            return jsonify({'error': 'No file selected'}), 400
        
        # Get file size
        file_size = get_file_size(file)
        logger.info(f"Document upload attempt - filename: {file.filename}, size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
        
        # Check file size limit
        if file_size > MAX_DOCUMENT_SIZE:
            logger.error(f"Document upload rejected - file size {file_size} exceeds limit of {MAX_DOCUMENT_SIZE} bytes")
            return jsonify({
                'error': 'File too large',
                'max_size': f'{MAX_DOCUMENT_SIZE / (1024*1024):.0f}MB',
                'uploaded_size': f'{file_size / (1024*1024):.2f}MB'
            }), 413
        
        # Check file extension
        if not allowed_file(file.filename, ALLOWED_DOCUMENT_EXTENSIONS):
            logger.warning(f"Document upload rejected - invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type', 'allowed': list(ALLOWED_DOCUMENT_EXTENSIONS)}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, 'document_' + filename)
        file.save(filepath)
        
        logger.info(f"Document uploaded successfully - {filename} ({file_size} bytes)")
        return jsonify({
            'success': True,
            'message': 'Document uploaded successfully',
            'filename': filename,
            'size': file_size
        }), 200
        
    except Exception as e:
        logger.exception(f"Error processing document upload: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/upload/media', methods=['POST'])
def upload_media():
    """
    Handle media file uploads
    Maximum size: 200MB
    Allowed formats: MP4, AVI, MOV, MP3, WAV, PNG, JPG, JPEG
    """
    try:
        logger.info("Media upload request received")
        
        if 'file' not in request.files:
            logger.warning("No file part in media upload request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            logger.warning("Empty filename in media upload request")
            return jsonify({'error': 'No file selected'}), 400
        
        # Get file size
        file_size = get_file_size(file)
        logger.info(f"Media upload attempt - filename: {file.filename}, size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
        
        # Check file size limit
        if file_size > MAX_MEDIA_SIZE:
            logger.error(f"Media upload rejected - file size {file_size} exceeds limit of {MAX_MEDIA_SIZE} bytes")
            return jsonify({
                'error': 'File too large',
                'max_size': f'{MAX_MEDIA_SIZE / (1024*1024):.0f}MB',
                'uploaded_size': f'{file_size / (1024*1024):.2f}MB'
            }), 413
        
        # Check file extension
        if not allowed_file(file.filename, ALLOWED_MEDIA_EXTENSIONS):
            logger.warning(f"Media upload rejected - invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type', 'allowed': list(ALLOWED_MEDIA_EXTENSIONS)}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, 'media_' + filename)
        file.save(filepath)
        
        logger.info(f"Media uploaded successfully - {filename} ({file_size} bytes)")
        return jsonify({
            'success': True,
            'message': 'Media uploaded successfully',
            'filename': filename,
            'size': file_size
        }), 200
        
    except Exception as e:
        logger.exception(f"Error processing media upload: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200


def main():
    """Main entry point for the application"""
    logger.info("Starting Flask upload application")
    logger.info(f"Avatar upload limit: {MAX_AVATAR_SIZE / (1024*1024):.0f}MB")
    logger.info(f"Document upload limit: {MAX_DOCUMENT_SIZE / (1024*1024):.0f}MB")
    logger.info(f"Media upload limit: {MAX_MEDIA_SIZE / (1024*1024):.0f}MB")
    
    # Run the application
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    main()