<?php
// Sample Page Component
// Version: 1.0
// Description: Displays a basic information page

$title = 'Sample Page';
$version = '1.0';
$author = 'Development Team';
$features = array('User Management', 'File Processing', 'Data Analytics');

echo '<html><head><title>' . $title . '</title></head><body>';
echo '<h1>' . htmlspecialchars($title) . '</h1>';
echo '<p>Version: ' . htmlspecialchars($version) . '</p>';
echo '<p>Author: ' . htmlspecialchars($author) . '</p>';
echo '<h2>Available Features:</h2><ul>';

// Display all available features
foreach ($features as $feature) {
    echo '<li>' . htmlspecialchars($feature) . '</li>';
}

echo '</ul></body></html>';
?>