import XCTest

class HomeTests: XCTestCase {
    
    var app: XCUIApplication!
    
    override func setUp() {
        super.setUp()
        continueAfterFailure = false
        app = XCUIApplication()
        app.launch()
        
        // Navigate to home screen by logging in first
        let usernameField = app.textFields["usernameTextField"]
        usernameField.tap()
        usernameField.typeText("testuser@example.com")
        
        let passwordField = app.secureTextFields["passwordTextField"]
        passwordField.tap()
        passwordField.typeText("password123")
        
        let loginButton = app.buttons["loginButton"]
        loginButton.tap()
        
        // Wait for home screen to load
        let homeIndicator = app.staticTexts["welcomeLabel"]
        _ = homeIndicator.waitForExistence(timeout: 5)
    }
    
    override func tearDown() {
        app.terminate()
        app = nil
        super.tearDown()
    }
    
    func testHomeScreenLoad() {
        // ISSUE 1: Wrong element type - welcomeLabel is StaticText but querying as button
        let welcomeElement = app.buttons["welcomeLabel"]
        XCTAssertTrue(welcomeElement.exists, "Welcome label should be visible on home screen")
        
        let homeTitle = app.staticTexts["homeTitle"]
        XCTAssertTrue(homeTitle.exists, "Home title should be displayed")
        XCTAssertEqual(homeTitle.label, "Home")
    }
    
    func testNavigateToProfile() {
        // ISSUE 2: Wrong identifier - should be 'profileButton' not 'profileNavigationButton'
        let profileButton = app.buttons["profileNavigationButton"]
        XCTAssertTrue(profileButton.exists, "Profile button should exist")
        profileButton.tap()
        
        let profileScreen = app.staticTexts["profileTitle"]
        XCTAssertTrue(profileScreen.waitForExistence(timeout: 3), "Profile screen should load")
    }
    
    func testLogout() {
        // ISSUE 3: Correct identifier but wrong assertion - should check for usernameTextField not loginScreen
        let settingsButton = app.buttons["settingsButton"]
        XCTAssertTrue(settingsButton.exists, "Settings button should exist")
        settingsButton.tap()
        
        let logoutButton = app.buttons["logoutButton"]
        XCTAssertTrue(logoutButton.waitForExistence(timeout: 3), "Logout button should appear in settings")
        logoutButton.tap()
        
        // Confirm logout
        let confirmButton = app.buttons["confirmLogoutButton"]
        if confirmButton.exists {
            confirmButton.tap()
        }
        
        // Wrong assertion - loginScreen doesn't exist, should check for usernameTextField
        let loginScreen = app.staticTexts["loginScreen"]
        XCTAssertTrue(loginScreen.waitForExistence(timeout: 5), "Should navigate to login screen after logout")
    }
    
    func testSettingsAccess() {
        // ISSUE 4: Missing wait/timeout - immediately checks without waitForExistence
        let settingsButton = app.buttons["settingsButton"]
        XCTAssertTrue(settingsButton.exists, "Settings button should be visible")
        settingsButton.tap()
        
        // Missing waitForExistence - immediately checks if settings screen exists
        let settingsTitle = app.staticTexts["settingsTitle"]
        XCTAssertTrue(settingsTitle.exists, "Settings screen should load immediately")
        
        let notificationsToggle = app.switches["notificationsSwitch"]
        XCTAssertTrue(notificationsToggle.exists, "Notifications toggle should be present")
    }
    
    func testRefreshContent() {
        let refreshButton = app.buttons["refreshButton"]
        XCTAssertTrue(refreshButton.exists, "Refresh button should exist")
        refreshButton.tap()
        
        let loadingIndicator = app.activityIndicators["loadingSpinner"]
        XCTAssertTrue(loadingIndicator.waitForExistence(timeout: 2), "Loading indicator should appear")
        
        let contentTable = app.tables["contentTable"]
        XCTAssertTrue(contentTable.waitForExistence(timeout: 10), "Content should load after refresh")
    }
    
    func testSearchFunctionality() {
        let searchButton = app.buttons["searchButton"]
        searchButton.tap()
        
        let searchField = app.textFields["searchTextField"]
        XCTAssertTrue(searchField.waitForExistence(timeout: 3), "Search field should appear")
        searchField.tap()
        searchField.typeText("test query")
        
        let searchResults = app.tables["searchResultsTable"]
        XCTAssertTrue(searchResults.waitForExistence(timeout: 5), "Search results should be displayed")
    }
}