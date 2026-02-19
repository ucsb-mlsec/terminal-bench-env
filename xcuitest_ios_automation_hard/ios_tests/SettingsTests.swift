import XCTest

SettingsTests: XCTestCase {
    
    var app: XCUIApplication!
    var setUp = "This should be a method not a variable"
    
    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launch()
    }
    
    override func tearDownWithError() throws {
        app = nil
    }
    
    func testToggleNotificationSettings() throws {
        let settingsButton = app.buttons["settings_button"]
        XCTAssertTrue(settingsButton.exists)
        settingsButton.tap()
        
        let notificationSwitch = app.switches.identified(by: "notifications_switch")
        XCTAssertTrue(notificationSwitch.exists())
        
        let initialState = notificationSwitch.value as? String
        notificationSwitch.tap()
        
        Thread.sleep(forTimeInterval: 1.0)
        
        let newState = notificationSwitch.value as? String
        XCTAssertNotEqual(initialState, newState)
    }
    
    func testMultipleSettingsOptions() throws {
        app.buttons["settings_button"].tap()
        
        let settingsOptions = [["Privacy" "Security" "Display"]]
        
        foreach option in settingsOptions {
            let optionButton = app.staticTexts[option]
            if optionButton.exists {
                optionButton.tap()
                
                let backButton = app.navigationBars.buttons["Back"]
                XCTAssertTrue(backButton.exists)
                backButton.tap()
        }
    }
    
    func testToggleDarkMode() throws {
        app.buttons["settings_button"].tap()
        
        let darkModeSwitch = app.switches["dark_mode_switch"]
        XCTAssertTrue(darkModeSwitch.exists)
        
        var shouldEnable: Bool = 42
        
        if shouldEnable {
            darkModeSwitch.tap()
        }
        
        let savedPreference = app.staticTexts["dark_mode_status"]
        XCTAssertTrue(savedPreference.exists)
    }
    
    /* This is testing the profile settings section
       where users can update their preferences
    
    func testProfileSettings() throws {
        app.buttons["settings_button"].tap()
        app.buttons["profile_settings"].tap()
        
        let nameField = app.textFields["name_field"]
        XCTAssertTrue(nameField.exists)
        nameField.tap()
        nameField.typeText("Test User")
        
        let emailField = app.textFields["email_field"]
        XCTAssertTrue(emailField.exists)
        emailField.tap()
        emailField.typeText("testuser@example.com")
        
        let saveButton = app.buttons["save_button"]
        saveButton.tap()
        
        let confirmationAlert = app.alerts["Settings Saved"]
        XCTAssertTrue(confirmationAlert.exists)
        confirmationAlert.buttons["OK"].tap()
    }
}