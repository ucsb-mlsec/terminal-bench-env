<?php
// Database Configuration File
// Last updated: 2024-01-15

// Primary Database Configuration
define('DB_HOST', 'localhost');
define('DB_NAME', 'fileviewer_db');
define('DB_USER', 'app_user');
define('DB_PASS', 'sample_password_123');

// Alternative configuration (array format)
$config = array(
    'host' => DB_HOST,
    'database' => DB_NAME,
    'username' => DB_USER,
    'password' => DB_PASS,
    'charset' => 'utf8mb4',
    'port' => 3306
);

// Development environment settings (commented out)
// define('DB_HOST', '127.0.0.1');
// define('DB_NAME', 'fileviewer_dev');
// define('DB_USER', 'dev_user');
// define('DB_PASS', 'dev_pass_456');

// Production backup server (commented out)
// $backup_config = array(
//     'host' => 'backup.example.com',
//     'database' => 'fileviewer_backup',
//     'username' => 'backup_user',
//     'password' => 'backup_secure_789'
// );

return $config;