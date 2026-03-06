#!/usr/bin/env python3

import asyncio
import websockets
import json
from collections import defaultdict
from time import time

# Store connected clients
connected_clients = set()

# Rate limiting: store message timestamps per user
user_message_times = defaultdict(list)

# Rate limit configuration
MAX_MESSAGES = 5
TIME_WINDOW = 10  # seconds

def check_rate_limit(username):
    """Check if user is within rate limit. Returns True if allowed, False if exceeded."""
    current_time = time()
    
    # Get user's message history
    message_times = user_message_times[username]
    
    # Remove timestamps older than TIME_WINDOW
    user_message_times[username] = [t for t in message_times if current_time - t < TIME_WINDOW]
    
    # Check if under limit
    if len(user_message_times[username]) < MAX_MESSAGES:
        user_message_times[username].append(current_time)
        return True
    return False

async def broadcast(message, exclude=None):
    """Broadcast message to all connected clients except the excluded one."""
    if connected_clients:
        tasks = []
        for client in connected_clients:
            if client != exclude:
                tasks.append(client.send(message))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

async def handler(websocket):
    """Handle individual client connections."""
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            try:
                # Parse incoming JSON message
                data = json.loads(message)
                username = data.get("user", "")
                
                # Check rate limit
                if check_rate_limit(username):
                    # Within rate limit - broadcast to all clients
                    await broadcast(message)
                else:
                    # Rate limit exceeded - send error to this client only
                    error_response = json.dumps({"error": "rate_limit_exceeded"})
                    await websocket.send(error_response)
                    
            except json.JSONDecodeError:
                # Invalid JSON - ignore or handle as needed
                pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.remove(websocket)

async def main():
    """Start the WebSocket server."""
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()  # Run forever

if __name__ == '__main__':
    asyncio.run(main())