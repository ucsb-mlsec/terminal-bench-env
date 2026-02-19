<?php
error_reporting(E_ALL);
ini_set('display_errors', 1);

// Legacy PHP Application - Session Manager
// WARNING: This code is for demonstration purposes only

require_once 'Logger.php';
require_once 'FileReader.php';

/**
 * Session Data Processor
 * Handles user session restoration from serialized data
 */

// Accept session data via GET parameter
if (isset($_GET['data'])) {
    try {
        // VULNERABILITY: Unsafe deserialization of user-controlled input
        // This allows attackers to inject arbitrary objects
        $session_data = unserialize($_GET['data']);
        
        if ($session_data) {
            echo "<div style='color: green;'>Session data processed successfully!</div>";
            echo "<pre>Session restored for user context</pre>";
        } else {
            echo "<div style='color: orange;'>Invalid session data format</div>";
        }
    } catch (Exception $e) {
        echo "<div style='color: red;'>Error processing session: " . htmlspecialchars($e->getMessage()) . "</div>";
    }
} else {
    // Display information about the application
    echo "<h2>Legacy Session Manager</h2>";
    echo "<p>This application handles user session data restoration.</p>";
    echo "<p>Session data should be provided via the 'data' GET parameter.</p>";
    
    // Example usage (for legitimate purposes)
    echo "<hr>";
    echo "<small>Example: index.php?data=&lt;serialized_session_data&gt;</small>";
}

?>

<!DOCTYPE html>
<html>
<head>
    <title>Session Manager</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Legacy Web Application</h1>
        <p>Session Management System v1.0</p>
    </div>
</body>
</html>