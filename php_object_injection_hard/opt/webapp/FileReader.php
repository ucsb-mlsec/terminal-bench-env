<?php

class FileReader {
    private $filename;
    
    public function __construct($filename = '') {
        $this->filename = $filename;
    }
    
    public function setFilename($filename) {
        $this->filename = $filename;
    }
    
    public function getFilename() {
        return $this->filename;
    }
    
    public function __toString() {
        if (file_exists($this->filename)) {
            return file_get_contents($this->filename);
        }
        return "File not found: " . $this->filename;
    }
}

?>