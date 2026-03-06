#!/usr/bin/env python3
# TODO: Add signature verification for X-Payment-Signature header

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/webhook/payment', methods=['POST'])
def payment_webhook():
    # No signature verification - security vulnerability!
    data = request.get_json()
    print(f"Processing payment: {data.get('amount')}")
    return jsonify({'status': 'processed'})

if __name__ == '__main__':
    app.run()