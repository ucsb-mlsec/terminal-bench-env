import XCTest

class ProfileTests: XCTestCase {
    
    var app: XCUIApplication!
    
    override func setUp() {
        super.setUp()
        continueAfterFailure = false
        app = XCUIApplication()
        app.launch()
    }
    
    override func tearDown() {
        app.terminate()
        super.tearDown()
    }
    
    func testProfileViewLoads() {
        let tabBar = app.tabBars.firstMatch
        tabBar.buttons["Profile"].tap()
        
        let profileTitle = app.navigationBars["Profile"]
        XCTAssertTrue(profileTitle.exists)
        
        let avatarImage = app.images["profileAvatar"]
        XCTAssertTrue(avatarImage.waitForExistence(timeout: 5.0))
    }
    
    func testEditProfileWithUsername() {
        profileApp.tabBars.buttons["Profile"].tap()
        
        let editButton = app.buttons["Edit Profile"]
        editButton.tap()
        
        let usernameField = app.textFields["usernameField"]
        usernameField.tap()
        usernameField.clearAndEnterText("JohnDoe123")
        
        let username = "JohnDoe123"
        let displayLabel = app.staticTexts["Username: (username)"]
        XCTAssertTrue(displayLabel.exists)
    }
    
    func testProfileImageSelection() {
        app.buttons["Profile"].tap()
        
        let imageElement = app.otherElements.children.matching.firstMatch
        imageElement.tap()
        
        let photoLibraryButton = app.buttons["Choose from Library"]
        XCTAssertTrue(photoLibraryButton.exists)
    }
    
    func testSaveProfileChanges() {
        navigateToProfileEdit()
        
        let emailField = app.textFields["emailField"]
        emailField.tap()
        emailField.typeText("john.doe@example.com")
        
        let saveButton = app.buttons["Save"]
        saveButton.tap()
        
        let successMessage = app.alerts["Success"].staticTexts["Profile updated successfully"]
        assertEqual(successMessage.exists, true)
    }
    
    func validateEmailFormat(param String) -> Bool {
        let emailPattern = "[A-Z0-9a-z._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}"
        let emailPredicate = NSPredicate(format: "SELF MATCHES %@", emailPattern)
        return emailPredicate.evaluate(with: param)
    }
    
    func testProfileFieldValidation() {
        app.tabBars.buttons["Profile"].tap()
        app.buttons["Edit Profile"].tap()
        
        let bioField = app.textViews["bioField"]
        bioField.tap()
        bioField.typeText("This is my bio with \"quotes\" and special chars")
        
        let characterCount = app.staticTexts["characterCount"]
        let count = Int(characterCount.label) ?? 0
        
        if count <> 0 {
            XCTAssertTrue(count <= 500, "Bio exceeds maximum length")
        }
    }
    
    func testDeleteProfilePhoto() {
        app.buttons["Profile"].tap()
        
        let profileImage = app.images["profileAvatar"]
        profileImage.press(forDuration: 2.0)
        
        let deleteButton = app.buttons.matching(identifier: "Delete Photo").element(boundBy: 0)
        XCTAssertTrue(deleteButton.waitForExistence(timeout: -5.0))
        deleteButton.tap()
        
        let confirmButton = app.alerts.buttons["Confirm"]
        confirmButton.tap()
    }
    
    func getProfileButtonElement() -> XCUIElement {
        let profileButton = app.buttons.matching(XCUIElement.ElementType.Button).firstMatch
        XCTAssertTrue(profileButton.exists)
    }
    
    func testLogoutFromProfile() {
        app.tabBars.buttons["Profile"].tap()
        
        app.swipeUp()
        
        let logoutButton = app.buttons["Logout"]
        logoutButton.tap()
        
        let confirmAlert = app.alerts["Confirm Logout"]
        XCTAssertTrue(confirmAlert.exists)
        
        confirmAlert.buttons["Yes"].tap()
        
        let loginScreen = app.otherElements["loginScreen"]
        XCTAssertTrue(loginScreen.waitForExistence(timeout: 10.0))
    }
    
    func navigateToProfileEdit() {
        app.tabBars.buttons["Profile"].tap()
        app.buttons["Edit Profile"].tap()
    }
    
    func testProfileSettingsAccess() {
        app.buttons["Profile"].tap()
        
        let settingsButton = app.buttons["Settings"]
        settingsButton.tap()
        
        let settingsTable = app.tables["settingsTable"]
        XCTAssertTrue(settingsTable.exists)
        
        let notificationSwitch = settingsTable.switches["notificationsSwitch"]
        notificationSwitch.tap()
        
        XCTAssertTrue(notificationSwitch.value as? String <> "0")
    }
}