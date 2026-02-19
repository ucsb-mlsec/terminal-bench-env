#!/usr/bin/env python3

from flask import Flask, jsonify, request, send_from_directory
import os

app = Flask(__name__)

# Product data - 50 products total
PRODUCTS = [
    {"id": "P001", "name": "Laptop", "price": 899.99},
    {"id": "P002", "name": "Mouse", "price": 24.50},
    {"id": "P003", "name": "Keyboard", "price": 79.99},
    {"id": "P004", "name": "Monitor", "price": 299.99},
    {"id": "P005", "name": "Webcam", "price": 59.99},
    {"id": "P006", "name": "Headphones", "price": 149.99},
    {"id": "P007", "name": "USB Cable", "price": 12.99},
    {"id": "P008", "name": "Desk Lamp", "price": 34.99},
    {"id": "P009", "name": "Phone Charger", "price": 19.99},
    {"id": "P010", "name": "Tablet", "price": 499.99},
    {"id": "P011", "name": "Wireless Router", "price": 89.99},
    {"id": "P012", "name": "External Hard Drive", "price": 119.99},
    {"id": "P013", "name": "USB Hub", "price": 29.99},
    {"id": "P014", "name": "Laptop Stand", "price": 45.99},
    {"id": "P015", "name": "Mousepad", "price": 14.99},
    {"id": "P016", "name": "Speakers", "price": 79.99},
    {"id": "P017", "name": "Microphone", "price": 99.99},
    {"id": "P018", "name": "HDMI Cable", "price": 15.99},
    {"id": "P019", "name": "Power Strip", "price": 24.99},
    {"id": "P020", "name": "Smartwatch", "price": 249.99},
    {"id": "P021", "name": "Fitness Tracker", "price": 69.99},
    {"id": "P022", "name": "Bluetooth Speaker", "price": 54.99},
    {"id": "P023", "name": "Gaming Mouse", "price": 49.99},
    {"id": "P024", "name": "Mechanical Keyboard", "price": 129.99},
    {"id": "P025", "name": "Gaming Headset", "price": 89.99},
    {"id": "P026", "name": "Graphics Tablet", "price": 199.99},
    {"id": "P027", "name": "Webcam Cover", "price": 9.99},
    {"id": "P028", "name": "Laptop Bag", "price": 39.99},
    {"id": "P029", "name": "Screen Protector", "price": 19.99},
    {"id": "P030", "name": "Phone Case", "price": 24.99},
    {"id": "P031", "name": "Wireless Earbuds", "price": 129.99},
    {"id": "P032", "name": "Portable Charger", "price": 44.99},
    {"id": "P033", "name": "SD Card", "price": 29.99},
    {"id": "P034", "name": "Card Reader", "price": 16.99},
    {"id": "P035", "name": "Ethernet Cable", "price": 11.99},
    {"id": "P036", "name": "Surge Protector", "price": 34.99},
    {"id": "P037", "name": "Desk Organizer", "price": 22.99},
    {"id": "P038", "name": "Monitor Arm", "price": 99.99},
    {"id": "P039", "name": "Ergonomic Mouse", "price": 64.99},
    {"id": "P040", "name": "Wrist Rest", "price": 18.99},
    {"id": "P041", "name": "Cable Management", "price": 14.99},
    {"id": "P042", "name": "Printer", "price": 199.99},
    {"id": "P043", "name": "Scanner", "price": 149.99},
    {"id": "P044", "name": "Stylus Pen", "price": 39.99},
    {"id": "P045", "name": "Drawing Tablet", "price": 299.99},
    {"id": "P046", "name": "Ring Light", "price": 49.99},
    {"id": "P047", "name": "Tripod", "price": 59.99},
    {"id": "P048", "name": "Phone Holder", "price": 19.99},
    {"id": "P049", "name": "Laptop Cooling Pad", "price": 34.99},
    {"id": "P050", "name": "Docking Station", "price": 179.99}
]

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/products')
def get_products():
    page = request.args.get('page', 1, type=int)
    
    # Calculate pagination
    per_page = 10
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    # Get products for the requested page
    products_page = PRODUCTS[start_idx:end_idx]
    
    return jsonify({
        'products': products_page,
        'page': page,
        'total_pages': 5,
        'total_products': 50
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)