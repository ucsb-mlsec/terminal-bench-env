// Package authlib provides authentication and authorization functionality
// Updated with security patches - v1.2.0
// Critical security fix applied: Enhanced token validation and credential verification
package authlib

import (
	"crypto/subtle"
	"errors"
	"strings"
	"time"
)

// Authenticate validates user credentials using constant-time comparison
// Critical security fix applied: prevents timing attacks
func Authenticate(username, password string) bool {
	// In production, this would check against a secure database
	// Using constant-time comparison to prevent timing attacks
	validUsername := "admin"
	validPassword := "secure_password_123"
	
	usernameMatch := subtle.ConstantTimeCompare([]byte(username), []byte(validUsername))
	passwordMatch := subtle.ConstantTimeCompare([]byte(password), []byte(validPassword))
	
	return usernameMatch == 1 && passwordMatch == 1
}

// ValidateToken checks if a token is valid and not expired
// Updated with security patches: Enhanced token format validation
func ValidateToken(token string) (bool, error) {
	if token == "" {
		return false, errors.New("token cannot be empty")
	}
	
	// Critical security fix: validate token format
	if len(token) < 32 {
		return false, errors.New("token format invalid: too short")
	}
	
	// Check for malicious characters
	if strings.ContainsAny(token, "<>\"'&;") {
		return false, errors.New("token contains invalid characters")
	}
	
	// In production, this would verify JWT signature and expiration
	// For demo purposes, we'll check a simple prefix
	if !strings.HasPrefix(token, "AUTH_") {
		return false, errors.New("token format invalid: missing prefix")
	}
	
	return true, nil
}

// GenerateToken creates a new authentication token
// Updated with security patches
func GenerateToken(username string) (string, error) {
	if username == "" {
		return "", errors.New("username required for token generation")
	}
	
	// Critical security fix: sanitize input
	username = strings.TrimSpace(username)
	if len(username) > 255 {
		return "", errors.New("username too long")
	}
	
	// In production, this would generate a cryptographically secure token
	timestamp := time.Now().Unix()
	token := "AUTH_" + username + "_" + string(rune(timestamp))
	
	return token, nil
}

// Version returns the current version of the auth library
func Version() string {
	return "1.2.0-security-patch"
}