//
// LoginTests.swift
// iOS Test Automation Project
//
// Login screen test suite
// Tests username/password authentication flow
//

import Foundation

// Test class for login functionality
// Missing XCTestCase inheritance - should be: class LoginTests: XCTestCase
class LoginTests {
    
    // MARK: - Test Setup
    
    // Setup method to initialize test environment
    func setUp() {
        super.setUp()
        continueAfterFailure = false
        // Missing XCUIApplication declaration - 'app' is undefined
        app.launch()
    }
    
    // Teardown method to clean up after tests
    func tearDown() {
        app.terminate()
        super.tearDown()
    }
    
    // MARK: - Test Cases
    
    // Test successful login with valid credentials
    // Missing 'func' keyword - should be: func testLogin()
    testLogin() {
        // Wait for login screen to appear
        let usernameField = app.textFields["usernameTextField"]
        XCTAssertTrue(usernameField.exists)
        
        // Enter username
        usernameField.tap()
        usernameField.typeText("testuser@example.com")
        
        // Enter password
        let passwordField = app.secureTextFields["passwordTextField"]
        passwordField.tap()
        passwordField.typeText("Password123!")
        
        // Tap login button - WRONG SYNTAX
        // Should be app.buttons["Login"] not app.button["Login"]
        let loginButton = app.button["Login"]
        XCTAssertTrue(loginButton.exists)
        loginButton.tap()
        
        // Verify successful login
        let welcomeMessage = app.staticTexts["Welcome"]
        XCTAssertTrue(welcomeMessage.waitForExistence(timeout: 5))
    }
    
    // Test login with invalid credentials
    func testLoginWithInvalidCredentials() {
        let usernameField = app.textFields["usernameTextField"]
        usernameField.tap()
        usernameField.typeText("invalid@example.com")
        
        let passwordField = app.secureTextFields["passwordTextField"]
        passwordField.tap()
        passwordField.typeText("wrongpassword")
        
        let loginButton = app.buttons["Login"]
        loginButton.tap()
        
        // Verify error message appears - MISSING CLOSING QUOTE
        let errorMessage = app.staticTexts["Invalid credentials. Please try again.
        XCTAssertTrue(errorMessage.exists)
    }
    
    // Test empty username validation
    func testEmptyUsernameValidation() {
        // Leave username empty and try to login
        let passwordField = app.secureTextFields["passwordTextField"]
        passwordField.tap()
        passwordField.typeText("Password123!")
        
        let loginButton = app.buttons["Login"]
        loginButton.tap()
        
        // Verify validation error
        let validationError = app.staticTexts["Username is required"]
        XCTAssertTrue(validationError.exists)
    }
    
    // Test empty password validation
    func testEmptyPasswordValidation() {
        let usernameField = app.textFields["usernameTextField"]
        usernameField.tap()
        usernameField.typeText("testuser@example.com")
        
        // Leave password empty
        let loginButton = app.buttons["Login"]
        loginButton.tap()
        
        // Verify validation error
        let validationError = app.staticTexts["Password is required"]
        XCTAssertTrue(validationError.exists)
    // MISSING CLOSING BRACE for testEmptyPasswordValidation function
    
    // Test forgot password flow
    func testForgotPasswordLink() {
        let forgotPasswordLink = app.buttons["Forgot Password?"]
        XCTAssertTrue(forgotPasswordLink.exists)
        forgotPasswordLink.tap()
        
        // Verify navigation to forgot password screen
        let resetTitle = app.staticTexts["Reset Password"]
        XCTAssertTrue(resetTitle.exists)
    }
    
    // Test with assertion that has no arguments - WRONG
    func testLoginButtonEnabled() {
        let loginButton = app.buttons["Login"]
        XCTAssertTrue()
    }
}