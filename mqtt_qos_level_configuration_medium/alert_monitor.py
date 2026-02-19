#!/usr/bin/env python3

import paho.mqtt.client as mqtt
import json
import time

# Load MQTT configuration
with open('../config/mqtt_config.json', 'r') as f:
    config = json.load(f)

# MQTT callback functions
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker successfully")
        # Subscribing to critical alerts with QoS=0 (INCORRECT - should be higher for reliability)
        # QoS 0: At most once delivery - messages can be lost
        client.subscribe("sensors/temperature/alerts", qos=0)
        print("Subscribed to sensors/temperature/alerts with QoS=0")
    else:
        print(f"Connection failed with code {rc}")

def on_message(client, userdata, msg):
    try:
        # Parse the incoming JSON message
        payload = json.loads(msg.payload.decode())
        
        sensor_id = payload.get('sensor_id', 'Unknown')
        temperature = payload.get('temperature', 'N/A')
        timestamp = payload.get('timestamp', time.time())
        
        # Print critical alert details
        print(f"CRITICAL ALERT: Sensor {sensor_id} reported {temperature}°C at {timestamp}")
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON message: {e}")
    except Exception as e:
        print(f"Error processing message: {e}")

# Create MQTT client instance
client = mqtt.Client()

# Assign callback functions
client.on_connect = on_connect
client.on_message = on_message

# Connect to MQTT broker
try:
    client.connect(config['broker_host'], config['broker_port'], config['keepalive'])
    print(f"Connecting to MQTT broker at {config['broker_host']}:{config['broker_port']}")
    
    # Start the network loop to process callbacks
    client.loop_forever()
    
except KeyboardInterrupt:
    print("\nDisconnecting from broker...")
    client.disconnect()
except Exception as e:
    print(f"Error connecting to broker: {e}")