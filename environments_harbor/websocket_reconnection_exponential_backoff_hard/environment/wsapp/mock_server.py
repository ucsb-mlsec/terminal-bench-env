#!/usr/bin/env python3

import asyncio
import websockets
import threading
import logging
from typing import Set, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockWSServer:
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.server = None
        self.loop = None
        self.thread = None
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.connection_count = 0
        self.running = False
        self._stop_event = None

    def start(self):
        """Start the WebSocket server in a background thread."""
        if self.running:
            logger.warning("Server is already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        
        # Wait for server to be ready
        import time
        time.sleep(0.5)
        logger.info(f"MockWSServer started on {self.host}:{self.port}")

    def _run_server(self):
        """Run the asyncio event loop in the background thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._stop_event = asyncio.Event()
        
        try:
            self.loop.run_until_complete(self._start_server())
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.loop.close()

    async def _start_server(self):
        """Start the WebSocket server and handle connections."""
        async with websockets.serve(self._handle_client, self.host, self.port):
            logger.info(f"Server listening on ws://{self.host}:{self.port}")
            await self._stop_event.wait()

    async def _handle_client(self, websocket, path):
        """Handle individual client connections."""
        self.connection_count += 1
        self.clients.add(websocket)
        client_id = id(websocket)
        logger.info(f"Client {client_id} connected (total connections: {self.connection_count})")

        try:
            async for message in websocket:
                logger.info(f"Received from client {client_id}: {message}")
                # Echo the message back to the client
                await websocket.send(message)
                logger.info(f"Echoed back to client {client_id}: {message}")
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} connection closed")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client {client_id} disconnected")

    def stop(self):
        """Stop the WebSocket server."""
        if not self.running:
            return

        logger.info("Stopping MockWSServer...")
        self.running = False

        if self.loop and self._stop_event:
            # Schedule the stop event in the server's event loop
            asyncio.run_coroutine_threadsafe(self._stop_event.set(), self.loop)

        # Disconnect all clients
        if self.loop:
            for client in list(self.clients):
                asyncio.run_coroutine_threadsafe(client.close(), self.loop)

        # Wait for thread to finish
        if self.thread:
            self.thread.join(timeout=2.0)

        logger.info("MockWSServer stopped")

    def disconnect_client(self, client_index: int = 0):
        """Force disconnect a specific client (default: first client)."""
        if not self.clients:
            logger.warning("No clients connected to disconnect")
            return

        clients_list = list(self.clients)
        if client_index >= len(clients_list):
            logger.warning(f"Client index {client_index} out of range")
            return

        client = clients_list[client_index]
        if self.loop:
            asyncio.run_coroutine_threadsafe(client.close(), self.loop)
            logger.info(f"Disconnected client at index {client_index}")

    def disconnect_all_clients(self):
        """Force disconnect all connected clients."""
        if self.loop:
            for client in list(self.clients):
                asyncio.run_coroutine_threadsafe(client.close(), self.loop)
            logger.info("Disconnected all clients")

    def get_connection_count(self) -> int:
        """Get the total number of connection attempts."""
        return self.connection_count

    def get_active_clients(self) -> int:
        """Get the number of currently active clients."""
        return len(self.clients)

    def reset_connection_count(self):
        """Reset the connection count to zero."""
        self.connection_count = 0


if __name__ == "__main__":
    # Simple test of the mock server
    server = MockWSServer()
    server.start()
    
    try:
        import time
        print("Server running. Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.stop()