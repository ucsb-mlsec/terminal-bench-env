import XCTest

class SettingsTests: XCTestCase {
    
    var app: XCUIApplication!
    
    override func setUp() {
        super.setUp()
        continueAfterFailure = false
        app = XCUIApplication()
        app.launch()
        
        // Navigate to settings screen
        let tabBar = app.tabBars.firstMatch
        tabBar.buttons["Settings"].tap()
    }
    
    override func tearDown() {
        app = nil
        super.tearDown()
    }
    
    func testNotificationToggle() {
        // Issue: references 'notificationToggle' but actual identifier is 'notificationSwitch'
        let notificationToggle = app.switches["notificationToggle"]
        
        XCTAssertTrue(notificationToggle.exists, "Notification toggle should exist")
        
        let initialState = notificationToggle.value as? String
        notificationToggle.tap()
        
        let newState = notificationToggle.value as? String
        XCTAssertNotEqual(initialState, newState, "Toggle state should change after tap")
    }
    
    func testThemeSelection() {
        // Issue: uses correct 'themePicker' identifier but assertion expects 'Dark Mode' when it should be 'dark'
        let themePicker = app.pickers["themePicker"]
        
        XCTAssertTrue(themePicker.waitForExistence(timeout: 3), "Theme picker should exist")
        
        themePicker.tap()
        
        let darkModeOption = app.buttons["Dark Mode"]
        darkModeOption.tap()
        
        // Incorrect assertion - expects 'Dark Mode' but actual value is 'dark'
        XCTAssertEqual(themePicker.value as? String, "Dark Mode", "Theme should be set to Dark Mode")
    }
    
    func testAboutSection() {
        // Issue: references 'aboutSection' which doesn't exist (should be 'aboutButton')
        let aboutSection = app.buttons["aboutSection"]
        
        XCTAssertTrue(aboutSection.exists, "About section should be visible")
        aboutSection.tap()
        
        let aboutTitle = app.navigationBars["About"]
        XCTAssertTrue(aboutTitle.waitForExistence(timeout: 2), "About screen should be displayed")
        
        // Verify version information
        let versionLabel = app.staticTexts["versionLabel"]
        XCTAssertTrue(versionLabel.exists, "Version label should be present")
    }
    
    func testSettingsLoad() {
        // This test works correctly with proper waits and correct identifiers
        let settingsTitle = app.navigationBars["Settings"]
        XCTAssertTrue(settingsTitle.waitForExistence(timeout: 5), "Settings screen should load")
        
        let notificationSwitch = app.switches["notificationSwitch"]
        XCTAssertTrue(notificationSwitch.waitForExistence(timeout: 3), "Notification switch should be visible")
        
        let themePicker = app.pickers["themePicker"]
        XCTAssertTrue(themePicker.exists, "Theme picker should be present")
        
        let aboutButton = app.buttons["aboutButton"]
        XCTAssertTrue(aboutButton.exists, "About button should be accessible")
        
        // Verify all elements are enabled
        XCTAssertTrue(notificationSwitch.isEnabled, "Notification switch should be enabled")
        XCTAssertTrue(aboutButton.isEnabled, "About button should be enabled")
    }
}