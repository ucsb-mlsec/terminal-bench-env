#!/usr/bin/env python3

import paho.mqtt.client as mqtt
import json
import time

# Load MQTT configuration
with open('../config/mqtt_config.json', 'r') as f:
    config = json.load(f)

# Message counter
message_count = 0
readings_log = []

def on_connect(client, userdata, flags, rc):
    """Callback when connected to MQTT broker"""
    if rc == 0:
        print("Data Logger connected to MQTT broker")
        # QoS=2 - Incorrectly configured (too high for routine data)
        # This causes unnecessary network overhead for non-critical readings
        client.subscribe("sensors/temperature/readings", qos=2)
        print("Subscribed to sensors/temperature/readings with QoS=2")
    else:
        print(f"Connection failed with code {rc}")

def on_message(client, userdata, msg):
    """Callback when message is received"""
    global message_count, readings_log
    
    try:
        # Parse JSON message
        payload = json.loads(msg.payload.decode())
        temperature = payload.get('temperature')
        sensor_id = payload.get('sensor_id')
        timestamp = payload.get('timestamp', time.time())
        
        # Append to in-memory log
        readings_log.append({
            'sensor_id': sensor_id,
            'temperature': temperature,
            'timestamp': timestamp
        })
        
        message_count += 1
        
        # Print periodic summaries every 10 messages
        if message_count % 10 == 0:
            print(f"Logged {message_count} routine readings")
            
    except json.JSONDecodeError as e:
        print(f"Error parsing message: {e}")
    except Exception as e:
        print(f"Error processing message: {e}")

# Create MQTT client
client = mqtt.Client(client_id="data_logger")

# Set callbacks
client.on_connect = on_connect
client.on_message = on_message

# Connect to broker
try:
    client.connect(
        config.get('broker', 'localhost'),
        config.get('port', 1883),
        config.get('keepalive', 60)
    )
    
    print("Starting data logger with QoS=2 for routine readings...")
    print("WARNING: Using QoS=2 for non-critical data causes unnecessary overhead")
    
    # Start network loop
    client.loop_forever()
    
except KeyboardInterrupt:
    print(f"\nShutting down. Total readings logged: {message_count}")
    client.disconnect()
except Exception as e:
    print(f"Error: {e}")