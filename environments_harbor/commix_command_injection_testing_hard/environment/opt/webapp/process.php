<?php
/**
 * Legacy File Processing System
 * Version 1.2.3
 * 
 * This script processes uploaded files and generates reports
 * WARNING: This is legacy code - needs security review
 */

// Configuration
define('UPLOAD_DIR', '/var/www/uploads/');
define('REPORT_DIR', '/var/www/reports/');
define('MAX_FILE_SIZE', 10485760); // 10MB

// Initialize variables
$error = '';
$success = '';
$output = '';

?>
<!DOCTYPE html>
<html>
<head>
    <title>File Processing System</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .error { color: red; }
        .success { color: green; }
        .output { background: #f0f0f0; padding: 10px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>File Processing System</h1>
    <p>Enter filename to process and generate report</p>

<?php

// Check if filename parameter is provided
if (isset($_GET['filename'])) {
    $filename = $_GET['filename'];
    
    // Basic validation - check for obvious malicious patterns
    // Block some dangerous characters
    if (strpos($filename, '..') !== false) {
        $error = "Invalid filename: parent directory access not allowed";
    } elseif (strpos($filename, '/') === 0) {
        $error = "Invalid filename: absolute paths not allowed";
    } else {
        // Sanitize filename - remove some special chars
        // This should be safe, right? We're blocking the dangerous stuff
        $clean_filename = str_replace(array('|', '&', '`', '$'), '', $filename);
        
        // Process the file using system commands
        // We need to check if file exists and get its info
        $command = "ls -la " . UPLOAD_DIR . $clean_filename . " 2>&1";
        
        // Execute command and capture output
        $file_info = shell_exec($command);
        
        if ($file_info) {
            $output .= "<h3>File Information:</h3>";
            $output .= "<pre>" . htmlspecialchars($file_info) . "</pre>";
            
            // Generate file statistics
            $stats_command = "wc -l " . UPLOAD_DIR . $clean_filename . " 2>&1";
            $stats = shell_exec($stats_command);
            
            if ($stats) {
                $output .= "<h3>File Statistics:</h3>";
                $output .= "<pre>" . htmlspecialchars($stats) . "</pre>";
                $success = "File processed successfully!";
            } else {
                $error = "Could not generate statistics for file";
            }
            
            // Log the processing
            $log_entry = date('Y-m-d H:i:s') . " - Processed file: " . $clean_filename . "\n";
            file_put_contents('/var/log/fileprocessing.log', $log_entry, FILE_APPEND);
            
        } else {
            $error = "File not found or cannot be accessed";
        }
    }
    
    // Display results
    if ($error) {
        echo "<div class='error'><strong>Error:</strong> " . htmlspecialchars($error) . "</div>";
    }
    
    if ($success) {
        echo "<div class='success'><strong>Success:</strong> " . htmlspecialchars($success) . "</div>";
    }
    
    if ($output) {
        echo "<div class='output'>" . $output . "</div>";
    }
}

?>

    <form method="GET" action="">
        <label for="filename">Filename to process:</label><br>
        <input type="text" id="filename" name="filename" size="50" 
               placeholder="example.txt" value="<?php echo isset($_GET['filename']) ? htmlspecialchars($_GET['filename']) : ''; ?>"><br><br>
        <input type="submit" value="Process File">
    </form>
    
    <hr>
    <p><small>Supported formats: .txt, .csv, .log, .dat</small></p>
    <p><small>Maximum file size: <?php echo MAX_FILE_SIZE / 1048576; ?>MB</small></p>

</body>
</html>