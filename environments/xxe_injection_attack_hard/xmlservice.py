#!/usr/bin/env python3

from flask import Flask, request, jsonify
import xml.etree.ElementTree as ET

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_xml():
    try:
        xml_data = request.data.decode('utf-8')
        
        # Basic input validation - block if BOTH patterns exist
        if 'doctype' in xml_data.lower() and 'file://' in xml_data:
            return jsonify({'error': 'Suspicious patterns detected'}), 400
        
        # Vulnerable XML parsing - allows external entities
        parser = ET.XMLParser()
        root = ET.fromstring(xml_data, parser=parser)
        
        # Extract data from XML elements
        results = []
        for item in root.findall('.//item'):
            if item.text:
                results.append(item.text)
        
        return jsonify({
            'status': 'success',
            'data': results
        })
        
    except ET.ParseError as e:
        return jsonify({'error': f'XML parsing error: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)