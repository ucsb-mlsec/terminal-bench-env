const express = require('express');
const path = require('path');
const fs = require('fs').promises;
const bodyParser = require('body-parser');

const app = express();

// Configuration object
const config = {
    port: process.env.PORT || 3000,
    dataDir: path.join(__dirname, 'data'),
    maxRequestSize: '10mb',
    environment: process.env.NODE_ENV || 'development',
    logLevel: 'info'
};

// Middleware setup
app.use(bodyParser.json({ limit: config.maxRequestSize }));
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static('public'));

// Initialize data directory
async function initializeDataDirectory() {
    try {
        await fs.access(config.dataDir);
    } catch (error) {
        await fs.mkdir(config.dataDir, { recursive: true });
        console.log('Data directory created successfully');
    }
}

// Read data from file
async function readDataFile(filename) {
    try {
        const filePath = path.join(config.dataDir, filename);
        const data = await fs.readFile(filePath, 'utf8');
        return JSON.parse(data);
    } catch (error) {
        console.error(`Error reading file ${filename}:`, error.message);
        return null;
    }
}

// Write data to file
async function writeDataFile(filename, data) {
    const filePath = path.join(config.dataDir, filename);
    await fs.writeFile(filePath, JSON.stringify(data, null, 2));
    return true;
}

// Route handlers
app.get('/', (req, res) => {
    res.json({ message: 'Project 2 API Server', version: '1.0.0', status: 'running' });
});

app.get('/api/projects', async (req, res) => {
    const projects = await readDataFile('projects.json') || [];
    res.json({ success: true, data: projects });
});

app.post('/api/projects', async (req, res) => {
    const newProject = req.body;
    const projects = await readDataFile('projects.json') || [];
    projects.push({ ...newProject, id: Date.now(), createdAt: new Date().toISOString() });
    await writeDataFile('projects.json', projects);
    res.status(201).json({ success: true, message: 'Project created' });
});

// Start server
initializeDataDirectory().then(() => {
    app.listen(config.port, () => {
        console.log(`Server running on port ${config.port} in ${config.environment} mode`);
    });
});

module.exports = app;