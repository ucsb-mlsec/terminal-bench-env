#!/usr/bin/env php
<?php
/**
 * File Viewer CLI Application
 * Processes file viewing requests from a request log
 * WARNING: This script contains vulnerabilities for educational purposes
 */

// Check if running from CLI
if (php_sapi_name() !== 'cli') {
    die("This script must be run from command line\n");
}

echo "=== File Viewer Application Starting ===\n";
echo "Reading requests from log file...\n\n";

// Configuration
define('BASE_DIR', '/home/challenge/files/');
define('LOG_FILE', '/home/challenge/requests.log');
define('SESSION_DIR', '/tmp/php_sessions/');

// Ensure session directory exists
if (!is_dir(SESSION_DIR)) {
    mkdir(SESSION_DIR, 0777, true);
}

/**
 * Session handler - creates and manages session files
 */
function handle_session($session_id, $data = array()) {
    $session_file = SESSION_DIR . 'sess_' . $session_id;
    
    // Read existing session data
    $session_data = array();
    if (file_exists($session_file)) {
        $content = file_get_contents($session_file);
        if ($content) {
            $session_data = unserialize($content);
            if (!is_array($session_data)) {
                $session_data = array();
            }
        }
    }
    
    // Merge new data with existing session data
    $session_data = array_merge($session_data, $data);
    
    // Write session file with user-controlled data
    file_put_contents($session_file, serialize($session_data));
    
    return $session_file;
}

/**
 * Parse query string from request
 */
function parse_request($request_line) {
    // Extract query string from request line
    // Format: GET /path?param1=value1&param2=value2
    if (preg_match('/^(\w+)\s+([^\s]+)/', $request_line, $matches)) {
        $method = $matches[1];
        $uri = $matches[2];
        
        $params = array();
        if (strpos($uri, '?') !== false) {
            list($path, $query) = explode('?', $uri, 2);
            parse_str($query, $params);
        }
        
        return array(
            'method' => $method,
            'uri' => $uri,
            'params' => $params
        );
    }
    
    return null;
}

/**
 * Process file viewing request
 * VULNERABLE: Does not properly sanitize file parameter
 */
function process_file_request($params) {
    echo "Processing file request...\n";
    
    // Handle session if session_id is provided
    if (isset($params['session_id'])) {
        $session_data = array();
        
        // Store user agent in session if provided
        if (isset($params['user_agent'])) {
            $session_data['user_agent'] = $params['user_agent'];
        }
        
        // Store any custom data in session
        if (isset($params['data'])) {
            $session_data['custom_data'] = $params['data'];
        }
        
        $session_file = handle_session($params['session_id'], $session_data);
        echo "Session file created/updated: $session_file\n";
    }
    
    // Process file parameter - VULNERABLE TO LFI
    if (isset($params['file'])) {
        $file = $params['file'];
        
        // Construct file path - allows directory traversal!
        $file_path = BASE_DIR . $file;
        
        echo "Attempting to access: $file_path\n";
        
        // Check if file exists
        if (file_exists($file_path)) {
            echo "File found! Processing...\n";
            echo "--- File Content Start ---\n";
            
            // If it's a PHP file, include it (DANGEROUS!)
            if (preg_match('/\.php$/i', $file_path)) {
                include($file_path);
            } else {
                // For other files, just read and display
                echo file_get_contents($file_path);
            }
            
            echo "\n--- File Content End ---\n";
        } else {
            echo "Error: File not found or access denied\n";
        }
    } else {
        echo "No file parameter specified\n";
    }
}

// Main execution loop
if (!file_exists(LOG_FILE)) {
    die("Error: Request log file not found at " . LOG_FILE . "\n");
}

$log_content = file(LOG_FILE, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);

if (empty($log_content)) {
    echo "No requests to process\n";
    exit(0);
}

echo "Processing " . count($log_content) . " requests...\n\n";

foreach ($log_content as $line_num => $request_line) {
    echo "========================================\n";
    echo "Request #" . ($line_num + 1) . ": $request_line\n";
    echo "========================================\n";
    
    $request = parse_request($request_line);
    
    if ($request) {
        process_file_request($request['params']);
    } else {
        echo "Invalid request format\n";
    }
    
    echo "\n";
}

echo "=== File Viewer Application Finished ===\n";