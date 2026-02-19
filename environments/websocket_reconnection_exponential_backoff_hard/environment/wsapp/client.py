#!/usr/bin/env python3

import json
import websocket
import threading
import time
import sys

class WSClient:
    def __init__(self, config_path='config.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.url = config['server_url']
        self.ws = None
        self.running = False
        
    def connect(self):
        try:
            self.ws = websocket.WebSocketApp(
                self.url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            self.running = True
            wst = threading.Thread(target=self.ws.run_forever)
            wst.daemon = True
            wst.start()
            time.sleep(1)
        except Exception as e:
            print(f"Connection failed: {e}")
            sys.exit(1)
    
    def on_message(self, ws, message):
        print(f"Received: {message}")
    
    def on_error(self, ws, error):
        print(f"Error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        print("Connection closed")
        self.running = False
    
    def on_open(self, ws):
        print("Connection opened")
    
    def send_message(self, message):
        if self.ws and self.running:
            try:
                self.ws.send(message)
                print(f"Sent: {message}")
            except Exception as e:
                print(f"Send failed: {e}")
        else:
            print("Not connected, message lost")
    
    def disconnect(self):
        if self.ws:
            self.running = False
            self.ws.close()
    
    def is_connected(self):
        return self.running

if __name__ == "__main__":
    client = WSClient()
    client.connect()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        client.disconnect()
        print("Client stopped")