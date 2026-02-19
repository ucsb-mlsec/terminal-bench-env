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
        app = nil
        super.tearDown()
    }
    
    // ISSUE: References 'userNameLabel' (camelCase) but actual identifier is 'nameLabel'
    func testProfileDisplay() {
        let tabBar = app.tabBars.firstMatch
        let profileTab = tabBar.buttons["profileTab"]
        XCTAssertTrue(profileTab.exists, "Profile tab should exist")
        profileTab.tap()
        
        // This identifier is wrong - should be 'nameLabel' not 'userNameLabel'
        let userName = app.staticTexts["userNameLabel"]
        XCTAssertTrue(userName.exists, "User name label should be displayed")
        
        let emailLabel = app.staticTexts["emailLabel"]
        XCTAssertTrue(emailLabel.exists, "Email label should be displayed")
        
        let profileImage = app.images["profileImage"]
        XCTAssertTrue(profileImage.exists, "Profile image should be displayed")
    }
    
    // ISSUE: Uses correct 'editButton' but then references non-existent 'cancelButton'
    func testEditProfile() {
        let tabBar = app.tabBars.firstMatch
        let profileTab = tabBar.buttons["profileTab"]
        profileTab.tap()
        
        let editButton = app.buttons["editButton"]
        XCTAssertTrue(editButton.waitForExistence(timeout: 5), "Edit button should exist")
        editButton.tap()
        
        let nameField = app.textFields["nameField"]
        XCTAssertTrue(nameField.exists, "Name field should be editable")
        nameField.tap()
        nameField.typeText("John Doe")
        
        // This button doesn't exist in the UI - should cause test to fail
        let cancelButton = app.buttons["cancelButton"]
        XCTAssertTrue(cancelButton.exists, "Cancel button should exist")
        cancelButton.tap()
    }
    
    // ISSUE: Wrong expected value - expects 'Profile saved successfully' but should be 'Changes saved'
    func testSaveProfile() {
        let tabBar = app.tabBars.firstMatch
        let profileTab = tabBar.buttons["profileTab"]
        profileTab.tap()
        
        let editButton = app.buttons["editButton"]
        XCTAssertTrue(editButton.waitForExistence(timeout: 5), "Edit button should exist")
        editButton.tap()
        
        let nameField = app.textFields["nameField"]
        nameField.tap()
        nameField.clearAndEnterText(text: "Jane Smith")
        
        let bioField = app.textViews["bioField"]
        bioField.tap()
        bioField.typeText("iOS Developer")
        
        let saveButton = app.buttons["saveButton"]
        XCTAssertTrue(saveButton.exists, "Save button should exist")
        saveButton.tap()
        
        let successMessage = app.staticTexts["successMessage"]
        XCTAssertTrue(successMessage.waitForExistence(timeout: 3), "Success message should appear")
        
        // Wrong expected value - actual message is 'Changes saved' not 'Profile saved successfully'
        XCTAssertEqual(successMessage.label, "Profile saved successfully", "Should show success message")
    }
    
    // ISSUE: No wait before assertion - timing issue
    func testProfileNavigation() {
        let tabBar = app.tabBars.firstMatch
        let profileTab = tabBar.buttons["profileTab"]
        profileTab.tap()
        
        let settingsButton = app.buttons["settingsButton"]
        XCTAssertTrue(settingsButton.exists, "Settings button should exist")
        settingsButton.tap()
        
        // Missing waitForExistence - direct assertion without waiting
        let settingsTitle = app.navigationBars["Settings"]
        XCTAssertTrue(settingsTitle.exists, "Settings screen should be displayed")
        
        let privacyCell = app.cells["privacyCell"]
        XCTAssertTrue(privacyCell.exists, "Privacy cell should exist")
    }
}

extension XCUIElement {
    func clearAndEnterText(text: String) {
        guard let stringValue = self.value as? String else {
            XCTFail("Tried to clear and enter text into a non string value")
            return
        }
        
        self.tap()
        
        let deleteString = String(repeating: XCUIKeyboardKey.delete.rawValue, count: stringValue.count)
        self.typeText(deleteString)
        self.typeText(text)
    }
}