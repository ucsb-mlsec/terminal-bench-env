#!/usr/bin/env python3
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/webhook/sync', methods=['POST'])
def sync_webhook():
    # WARNING: No signature verification (X-Sync-Signature)
    # WARNING: No timestamp validation (X-Sync-Timestamp)
    
    # Directly processing without any verification
    data = request.get_json()
    
    # Process the sync without checking authenticity
    print(f"Syncing data: {data.get('records')} records")
    
    # Return success without validation
    return jsonify({'status': 'synced'})

if __name__ == '__main__':
    app.run(port=5002)