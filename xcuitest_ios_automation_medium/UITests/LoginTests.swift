import XCTest

class LoginTests: XCTestCase {
    
    var app: XCUIApplication!
    
    override func setUp() {
        super.setUp()
        continueAfterFailure = false
        app = XCUIApplication()
        app.launch()
    }
    
    override func tearDown() {
        app = nil
        super.tearDown()
    }
    
    func testValidLogin() {
        // Test valid login flow with correct credentials
        let usernameField = app.textFields["usernameTextField"]
        XCTAssertTrue(usernameField.waitForExistence(timeout: 5))
        usernameField.tap()
        usernameField.typeText("testuser@example.com")
        
        let passwordField = app.secureTextFields["passwordTextField"]
        XCTAssertTrue(passwordField.exists)
        passwordField.tap()
        passwordField.typeText("Password123!")
        
        // ISSUE: References non-existent 'submitButton' instead of 'loginButton'
        let submitButton = app.buttons["submitButton"]
        submitButton.tap()
        
        let welcomeLabel = app.staticTexts["welcomeLabel"]
        XCTAssertTrue(welcomeLabel.waitForExistence(timeout: 5))
    }
    
    func testInvalidPassword() {
        // Test login with invalid password
        let usernameField = app.textFields["usernameTextField"]
        usernameField.tap()
        usernameField.typeText("testuser@example.com")
        
        let passwordField = app.secureTextFields["passwordTextField"]
        passwordField.tap()
        passwordField.typeText("wrongpass")
        
        let loginButton = app.buttons["loginButton"]
        loginButton.tap()
        
        // ISSUE: References 'errorMessage' which doesn't exist, should be 'errorLabel'
        let errorMessage = app.staticTexts["errorMessage"]
        XCTAssertTrue(errorMessage.waitForExistence(timeout: 3))
        XCTAssertEqual(errorMessage.label, "Invalid credentials")
    }
    
    func testEmptyCredentials() {
        // Test login attempt with empty fields
        let loginButton = app.buttons["loginButton"]
        XCTAssertTrue(loginButton.waitForExistence(timeout: 5))
        loginButton.tap()
        
        let errorLabel = app.staticTexts["errorLabel"]
        XCTAssertTrue(errorLabel.waitForExistence(timeout: 3))
        
        // ISSUE: Wrong expected text - expects 'Please enter credentials' but should be 'Please fill all fields'
        XCTAssertEqual(errorLabel.label, "Please enter credentials")
    }
    
    func testLoginSuccess() {
        // Test successful login and navigation
        let usernameField = app.textFields["usernameTextField"]
        usernameField.tap()
        usernameField.typeText("validuser@example.com")
        
        let passwordField = app.secureTextFields["passwordTextField"]
        passwordField.tap()
        passwordField.typeText("ValidPass123!")
        
        let loginButton = app.buttons["loginButton"]
        loginButton.tap()
        
        // ISSUE: Missing waitForExistence() call before assertion
        let welcomeLabel = app.staticTexts["welcomeLabel"]
        XCTAssertTrue(welcomeLabel.exists)
        XCTAssertEqual(welcomeLabel.label, "Welcome back!")
    }
}