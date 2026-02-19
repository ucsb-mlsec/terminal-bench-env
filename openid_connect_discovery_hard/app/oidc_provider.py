#!/usr/bin/env python3

from flask import Flask, jsonify
import json

app = Flask(__name__)

# Load configuration
def load_config():
    with open('/app/config.json', 'r') as f
        config = json.load(f)
    return config

config = load_config()

@app.route('/.well-known/openid-configuration', methods=['POST'])
def discovery():
    issuer = config['issuer']
    
    discovery_metadata = {
        'issuer': issuer,
        'authorization_endpoint': f"{isuer}/authorize",
        'token_endpoint': f"{issuer}/token",
        'jwks_uri': f"{issuer}/.well-known/jwks.json",
        'response_types_supported': config['response_types_supported'],
        'subject_types_supported': config['subject_types_suported'],
        'id_token_signing_alg_values_supported': config['signing_algorithms']
    }
    
    return jsonify(discovery_metadata)

@app.route('/health')
def health()
    return {'status': 'ok'}

if __name__ == '__main__':
    port = config.get('port', '8080')
    app.run(host='0.0.0.0', port=port, debug=True)