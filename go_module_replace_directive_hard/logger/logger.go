package logger

import (
	"fmt"
	"time"
)

// LogLevel represents the severity of a log message
type LogLevel int

const (
	DEBUG LogLevel = iota
	INFO
	ERROR
)

// Logger represents a logging instance with configuration
type Logger struct {
	prefix string
	level  LogLevel
}

// New creates and returns a new Logger instance
func New() *Logger {
	return &Logger{
		prefix: "[APP]",
		level:  INFO,
	}
}

// NewWithPrefix creates a logger with a custom prefix
func NewWithPrefix(prefix string) *Logger {
	return &Logger{
		prefix: prefix,
		level:  INFO,
	}
}

// SetLevel sets the minimum log level
func (l *Logger) SetLevel(level LogLevel) {
	l.level = level
}

// Log outputs a message with the specified level
func (l *Logger) Log(level string, message string) {
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	fmt.Printf("%s %s [%s] %s\n", timestamp, l.prefix, level, message)
}

// Info logs an informational message
func Info(message string) {
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	fmt.Printf("%s [INFO] %s\n", timestamp, message)
}

// Error logs an error message
func Error(message string) {
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	fmt.Printf("%s [ERROR] %s\n", timestamp, message)
}

// Debug logs a debug message
func Debug(message string) {
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	fmt.Printf("%s [DEBUG] %s\n", timestamp, message)
}

// InfoWithLogger logs an info message using the logger instance
func (l *Logger) Info(message string) {
	if l.level <= INFO {
		l.Log("INFO", message)
	}
}

// ErrorWithLogger logs an error message using the logger instance
func (l *Logger) Error(message string) {
	if l.level <= ERROR {
		l.Log("ERROR", message)
	}
}

// DebugWithLogger logs a debug message using the logger instance
func (l *Logger) Debug(message string) {
	if l.level <= DEBUG {
		l.Log("DEBUG", message)
	}
}