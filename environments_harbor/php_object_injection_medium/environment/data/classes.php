<?php

class User {
    public $username;
    public $email;
    public $role;
    
    public function __construct($username, $email, $role = 'user') {
        $this->username = $username;
        $this->email = $email;
        $this->role = $role;
    }
    
    public function isAdmin() {
        return $this->role === 'admin';
    }
    
    public function getProfile() {
        return array(
            'username' => $this->username,
            'email' => $this->email,
            'role' => $this->role
        );
    }
}

class Session {
    public $sessionId;
    public $userId;
    public $lastActivity;
    
    public function __construct($sessionId, $userId) {
        $this->sessionId = $sessionId;
        $this->userId = $userId;
        $this->lastActivity = time();
    }
    
    public function isValid() {
        return (time() - $this->lastActivity) < 3600;
    }
    
    public function refresh() {
        $this->lastActivity = time();
    }
}

class AdminAction {
    private $action;
    private $target;
    private $adminUser;
    
    public function __construct($action = '', $target = '') {
        $this->action = $action;
        $this->target = $target;
    }
    
    public function setAction($action) {
        $this->action = $action;
    }
    
    public function setTarget($target) {
        $this->target = $target;
    }
    
    public function execute() {
        if ($this->action === 'delete_user') {
            $this->deleteUser($this->target);
        } elseif ($this->action === 'modify_privileges') {
            $this->modifyPrivileges($this->target);
        } elseif ($this->action === 'extract_data') {
            $this->extractData($this->target);
        }
    }
    
    private function deleteUser($username) {
        error_log("Deleting user: " . $username);
    }
    
    private function modifyPrivileges($username) {
        error_log("Modifying privileges for: " . $username);
    }
    
    private function extractData($table) {
        error_log("Extracting data from: " . $table);
    }
}

class Logger {
    public $logFile;
    public $logData;
    public $adminAction;
    
    public function __construct($logFile = '/tmp/app.log') {
        $this->logFile = $logFile;
        $this->logData = '';
    }
    
    public function setLogData($data) {
        $this->logData = $data;
    }
    
    public function __destruct() {
        if ($this->adminAction instanceof AdminAction) {
            $this->adminAction->execute();
        }
        
        if (!empty($this->logData)) {
            file_put_contents($this->logFile, $this->logData, FILE_APPEND);
        }
    }
}

class Profile {
    public $userId;
    public $displayName;
    public $bio;
    public $avatar;
    
    public function __construct($userId, $displayName) {
        $this->userId = $userId;
        $this->displayName = $displayName;
        $this->bio = '';
        $this->avatar = 'default.png';
    }
    
    public function updateBio($bio) {
        $this->bio = htmlspecialchars($bio);
    }
    
    public function updateAvatar($avatar) {
        $this->avatar = basename($avatar);
    }
}

?>