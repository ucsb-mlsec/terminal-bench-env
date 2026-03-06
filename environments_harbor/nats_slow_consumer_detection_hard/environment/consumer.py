#!/usr/bin/env python3

import asyncio
import time
import sys
from nats.aio.client import Client as NATS
from nats.errors import TimeoutError, ConnectionClosedError

# This consumer application experiences 'Slow Consumer' disconnections in production
# when processing bursts of large messages due to insufficient server configuration

async def main():
    """
    Main consumer application that subscribes to NATS and processes messages.
    This consumer is designed to handle:
    - Bursts of up to 50,000 pending messages
    - Large message payloads (up to 2MB)
    - Processing time of 2-5 seconds per message
    
    However, it frequently experiences 'Slow Consumer' disconnections
    due to insufficient server configuration parameters.
    """
    nc = NATS()
    
    # Statistics tracking
    messages_processed = 0
    disconnection_count = 0
    
    async def disconnected_cb():
        nonlocal disconnection_count
        disconnection_count += 1
        print(f"⚠️  DISCONNECTED from NATS server! (Count: {disconnection_count})")
        print("   Reason: Likely 'Slow Consumer' - pending messages exceeded server limits")
    
    async def reconnected_cb():
        print(f"✓ RECONNECTED to NATS server")
    
    async def error_cb(e):
        print(f"❌ ERROR: {e}")
        if "slow consumer" in str(e).lower():
            print("   SLOW CONSUMER DETECTED - Server disconnected due to too many pending messages")
    
    async def closed_cb():
        print("🔌 Connection CLOSED")
    
    async def message_handler(msg):
        """
        Message handler with simulated CPU-intensive processing.
        Processes each message with a 2-5 second delay, which can cause
        messages to queue up during high-throughput bursts.
        """
        nonlocal messages_processed
        
        subject = msg.subject
        data_size = len(msg.data)
        messages_processed += 1
        
        print(f"\n📨 Message #{messages_processed} received:")
        print(f"   Subject: {subject}")
        print(f"   Size: {data_size:,} bytes ({data_size/1024/1024:.2f} MB)")
        print(f"   Pending messages in queue: ~{messages_processed % 100}")
        
        # Simulate CPU-intensive processing (2-5 seconds)
        # This represents real-world processing like:
        # - Database operations
        # - Data transformation
        # - External API calls
        # - Complex business logic
        processing_time = 3.0  # 3 seconds per message
        print(f"   Processing message (will take {processing_time}s)...")
        
        # Simulate blocking work
        await asyncio.sleep(processing_time)
        
        print(f"   ✓ Message #{messages_processed} processed successfully")
    
    try:
        # Connect to NATS server with callbacks
        print("Connecting to NATS server at nats://localhost:4222...")
        await nc.connect(
            servers=["nats://localhost:4222"],
            disconnected_cb=disconnected_cb,
            reconnected_cb=reconnected_cb,
            error_cb=error_cb,
            closed_cb=closed_cb,
            max_reconnect_attempts=10,
            reconnect_time_wait=2
        )
        
        print("✓ Connected to NATS server")
        print("\nSubscribing to subject: 'events.data'")
        print("This consumer expects to handle:")
        print("  - Bursts of up to 50,000 pending messages")
        print("  - Message payloads up to 2MB in size")
        print("  - Processing time of ~3 seconds per message")
        print("\n⚠️  WARNING: Current server configuration causes frequent disconnections!")
        print("=" * 70)
        
        # Subscribe to the subject with pending limits
        # Note: These limits should be configured on the server side
        subscription = await nc.subscribe("events.data", cb=message_handler)
        
        print(f"\n✓ Subscribed to 'events.data' (SID: {subscription._id})")
        print("Waiting for messages... (Press Ctrl+C to stop)\n")
        
        # Keep the consumer running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Shutdown requested by user")
    except ConnectionClosedError as e:
        print(f"\n❌ Connection closed error: {e}")
        print("   This typically indicates a 'Slow Consumer' disconnection")
    except TimeoutError as e:
        print(f"\n❌ Timeout error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print(f"\n📊 Final Statistics:")
        print(f"   Messages processed: {messages_processed}")
        print(f"   Disconnections: {disconnection_count}")
        
        if nc.is_connected:
            print("\nClosing connection...")
            await nc.drain()
            await nc.close()
        
        print("Consumer stopped.")

if __name__ == '__main__':
    print("=" * 70)
    print("NATS Consumer Application - Slow Consumer Test")
    print("=" * 70)
    asyncio.run(main())