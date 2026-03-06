<?php

class Logger {
    public $logfile;
    
    public function __construct($file = null) {
        if ($file !== null) {
            $this->logfile = $file;
        }
    }
    
    public function log($message) {
        if ($this->logfile) {
            file_put_contents($this->logfile, $message . "\n", FILE_APPEND);
        }
    }
    
    public function __destruct() {
        if ($this->logfile) {
            $reader = new FileReader($this->logfile);
            echo $reader->getContent();
        }
    }
}

?>