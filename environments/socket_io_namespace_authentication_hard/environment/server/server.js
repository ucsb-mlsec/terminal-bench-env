const { Server } = require('socket.io');
const http = require('http');
const jwt = require('jsonwebtoken');

const httpServer = http.createServer();
const io = new Server(httpServer, {
  cors: {
    origin: '*',
    methods: ['GET', 'POST']
  }
});

const JWT_SECRET = process.env.JWT_SECRET || 'test-secret-key';

// TODO: Add authentication middleware
// Currently all namespaces accept connections without authentication - SECURITY VULNERABILITY!

// Public namespace - should be accessible to everyone
const publicNamespace = io.of('/public');
publicNamespace.on('connection', (socket) => {
  console.log('Client connected to /public');
  
  socket.on('disconnect', () => {
    console.log('Client disconnected from /public');
  });
});

// Admin namespace - should only be accessible to admin users
// TODO: Add admin role verification
const adminNamespace = io.of('/admin');
adminNamespace.on('connection', (socket) => {
  console.log('Admin connected');
  
  socket.emit('admin-data', 'Sensitive admin information');
  
  socket.on('disconnect', () => {
    console.log('Admin disconnected');
  });
});

// Analytics namespace - should only be accessible to analysts and admins
// TODO: Add analyst/admin role verification
const analyticsNamespace = io.of('/analytics');
analyticsNamespace.on('connection', (socket) => {
  console.log('Analyst connected');
  
  socket.emit('analytics-data', 'Analytics dashboard data');
  
  socket.on('disconnect', () => {
    console.log('Analyst disconnected');
  });
});

httpServer.listen(3000, () => {
  console.log('Server running on port 3000');
});