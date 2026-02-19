const WebSocket = require('ws');

let ws;
let reconnectAttempts = 0;
let reconnectDelay = 1000; // Start with 1 second
const MAX_RECONNECT_ATTEMPTS = 10;
const MAX_RECONNECT_DELAY = 30000; // 30 seconds
const INITIAL_RECONNECT_DELAY = 1000; // 1 second

function connect() {
  console.log('Attempting to connect to ws://localhost:8080...');
  
  ws = new WebSocket('ws://localhost:8080');
  
  ws.on('open', () => {
    console.log('Connected to WebSocket server');
    // Reset reconnection tracking on successful connection
    reconnectAttempts = 0;
    reconnectDelay = INITIAL_RECONNECT_DELAY;
  });
  
  ws.on('message', (data) => {
    console.log('Received message:', data.toString());
  });
  
  ws.on('close', () => {
    console.log('Connection closed');
    handleReconnect();
  });
  
  ws.on('error', (error) => {
    console.log('WebSocket error:', error.message);
    // Don't call handleReconnect here as 'close' event will fire after error
  });
}

function handleReconnect() {
  reconnectAttempts++;
  
  if (reconnectAttempts > MAX_RECONNECT_ATTEMPTS) {
    console.log(`Max reconnection attempts (${MAX_RECONNECT_ATTEMPTS}) reached. Exiting.`);
    process.exit(0);
    return;
  }
  
  console.log(`Reconnection attempt ${reconnectAttempts} in ${reconnectDelay}ms...`);
  
  setTimeout(() => {
    connect();
  }, reconnectDelay);
  
  // Exponential backoff: double the delay for next time
  reconnectDelay = Math.min(reconnectDelay * 2, MAX_RECONNECT_DELAY);
}

// Start the initial connection
connect();