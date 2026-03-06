package main

import (
	"fmt"
	"github.com/company/authlib"
)

func main() {
	fmt.Println("Starting main application...")
	
	// Test the authentication library
	token := "test-token-12345"
	
	// Call ValidateToken from the authlib package
	isValid, err := authlib.ValidateToken(token)
	if err != nil {
		fmt.Printf("Error validating token: %v\n", err)
		return
	}
	
	if isValid {
		fmt.Println("Authentication successful! Token is valid.")
	} else {
		fmt.Println("Authentication failed! Token is invalid.")
	}
	
	// Call Authenticate function to test additional functionality
	user, err := authlib.Authenticate("admin", "password123")
	if err != nil {
		fmt.Printf("Error authenticating user: %v\n", err)
		return
	}
	
	fmt.Printf("User authenticated: %s\n", user)
	fmt.Println("Application completed successfully with local authlib!")
}