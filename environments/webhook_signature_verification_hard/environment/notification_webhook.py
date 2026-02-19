#!/usr/bin/env python3

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/webhook/notification', methods=['POST'])
def notification_webhook():
    # FIXME: Missing HMAC-SHA512 verification for X-Notify-Sig
    data = request.get_json()
    print(f"Processing notification: {data.get('event')}")
    return jsonify({'status': 'received'})

if __name__ == '__main__':
    app.run(port=5001)