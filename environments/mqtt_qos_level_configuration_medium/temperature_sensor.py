#!/usr/bin/env python3

import paho.mqtt.client as mqtt
import json
import time
import random
import os

# MQTT callback functions
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker successfully")
    else:
        print(f"Failed to connect, return code {rc}")

def on_publish(client, userdata, mid):
    print(f"Message {mid} published")

# Load MQTT configuration
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'mqtt_config.json')
with open(config_path, 'r') as f:
    mqtt_config = json.load(f)

# Initialize MQTT client
client = mqtt.Client(client_id="temperature_sensor_001")
client.on_connect = on_connect
client.on_publish = on_publish

# Connect to broker
client.connect(mqtt_config['broker_host'], mqtt_config['broker_port'], 60)
client.loop_start()

# Give time to establish connection
time.sleep(1)

def publish_routine_reading(temperature):
    """
    Publishes routine temperature readings
    QoS=0: Fire and forget - messages may be lost (currently used incorrectly for all messages)
    """
    message = {
        "sensor_id": "SENSOR_001",
        "temperature": temperature,
        "timestamp": int(time.time()),
        "type": "routine"
    }
    
    # QoS level 0 - At most once delivery (no guarantee)
    qos = 0
    result = client.publish(
        mqtt_config['topics']['temperature_readings'],
        json.dumps(message),
        qos=qos
    )
    print(f"Published routine reading: {temperature}°C with QoS={qos}")
    return result

def publish_critical_alert(temperature):
    """
    Publishes critical temperature alerts
    PROBLEM: QoS=0 is used here, causing critical alerts to be lost!
    This should use higher QoS for guaranteed delivery
    """
    message = {
        "sensor_id": "SENSOR_001",
        "temperature": temperature,
        "timestamp": int(time.time()),
        "type": "critical"
    }
    
    # QoS level 0 - INCORRECTLY CONFIGURED! Critical alerts need guaranteed delivery
    qos = 0
    result = client.publish(
        mqtt_config['topics']['temperature_alerts'],
        json.dumps(message),
        qos=qos
    )
    print(f"!!! CRITICAL ALERT !!! Temperature: {temperature}°C published with QoS={qos} (PROBLEM: should use higher QoS)")
    return result

# Main loop
print("Temperature sensor starting...")
print("Publishing routine readings every 2 seconds")
print("Will randomly generate critical alerts (10% chance)")

try:
    while True:
        # Generate temperature reading
        # 90% chance: normal temperature (18-25°C)
        # 10% chance: critical temperature (>30°C)
        if random.random() < 0.1:
            # Critical temperature
            temperature = round(random.uniform(30.5, 40.0), 2)
            publish_critical_alert(temperature)
        else:
            # Routine temperature
            temperature = round(random.uniform(18.0, 25.0), 2)
            publish_routine_reading(temperature)
        
        time.sleep(2)

except KeyboardInterrupt:
    print("\nStopping temperature sensor...")
    client.loop_stop()
    client.disconnect()