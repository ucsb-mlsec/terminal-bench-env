#!/usr/bin/env python3

import asyncio
import json
import websockets

async def spam_test():
    uri = "ws://localhost:8765"
    
    alice_broadcast_count = 0
    bob_broadcast_count = 0
    errors_received_count = 0
    
    async with websockets.connect(uri) as websocket:
        # Send 20 messages from alice rapidly (within 2 seconds)
        alice_messages_sent = []
        for i in range(1, 21):
            message = {"user": "alice", "text": f"Message {i}"}
            await websocket.send(json.dumps(message))
            alice_messages_sent.append(f"Message {i}")
            await asyncio.sleep(0.1)  # Small delay, total ~2 seconds for 20 messages
        
        # Send 3 messages from bob
        bob_messages_sent = []
        for i in range(1, 4):
            message = {"user": "bob", "text": f"Bob message {i}"}
            await websocket.send(json.dumps(message))
            bob_messages_sent.append(f"Bob message {i}")
            await asyncio.sleep(0.1)
        
        # Receive responses for a sufficient time
        # We expect 5 alice broadcasts + 3 bob broadcasts + 15 errors = 23 messages
        # Add some buffer time
        try:
            for _ in range(30):  # Try to receive up to 30 messages with timeout
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                    data = json.loads(response)
                    
                    # Check if it's an error response
                    if "error" in data:
                        if data["error"] == "rate_limit_exceeded":
                            errors_received_count += 1
                    # Check if it's a broadcast message
                    elif "user" in data and "text" in data:
                        if data["user"] == "alice":
                            alice_broadcast_count += 1
                        elif data["user"] == "bob":
                            bob_broadcast_count += 1
                except asyncio.TimeoutError:
                    # No more messages received, break out
                    break
        except Exception as e:
            print(f"Error receiving messages: {e}")
    
    # Print results
    print(f"ALICE_BROADCAST={alice_broadcast_count}")
    print(f"BOB_BROADCAST={bob_broadcast_count}")
    print(f"ERRORS_RECEIVED={errors_received_count}")
    
    return alice_broadcast_count, bob_broadcast_count, errors_received_count

if __name__ == "__main__":
    asyncio.run(spam_test())