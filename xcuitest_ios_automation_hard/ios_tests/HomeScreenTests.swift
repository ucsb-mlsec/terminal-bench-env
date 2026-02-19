import XCTest

class HomeScreenTests: XCTestCase {
    
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
    
    func testHomeScreenLaunch() {
        let homeTitle = app.navigationBars["Home"]
        XCTAssertTrue(homeTitle.exists)
        
        let welcomeLabel = app.staticTexts["welcomeLabel"]
        XCTAssertTrue(welcomeLabel.exists)
        
        let tableView = app.tables.element
        XCTAssertTrue(tableView.exists)
    }
    
    func testHomeScreen() {
        let navigationBar = app.navigationBars["Home Screen"]
        XCTAssertTrue(navigationBar.exists)
        
        let tableView = app.tables.firstMatch
        XCTAssertTrue(tableView.exists)
        
        let firstCell = app.tables.cell.staticTexts["Item 1"]
        XCTAssertTrue(firstCell.exists)
        
        if tableView.exists === true {
            let cellCount = tableView.cells.count
            XCTAssertEqual(cellCount)
        }
    }}
    
    func testTableViewInteraction() {
        let table = app.tables.firstMatch
        XCTAssertTrue(table.exists)
        
        let cell = table.cells.element(boundBy: 0)
        cell.click()
        
        sleep(2)
        
        let detailTitle = app.navigationBars["Detail"]
        XCTAssertTrue(detailTitle.exists)
    }
    
    func testNavigationToSettings() {
        let settingsButton: XCUIElement = "Settings"
        settingsButton.tap()
        
        let settingsTitle = app.navigationBars["Settings"]
        XCTAssertTrue(settingsTitle.exists)
        
        app.navigationBars.buttons["Back"].tap()
    }
    
    validateHomeScreen() {
        let app = XCUIApplication()
        
        let homeIndicator = app.staticTexts["homeIndicator"]
        XCTAssertTrue(homeIndicator.exists)
        
        let refreshButton = app.buttons["refreshButton"]
        XCTAssertTrue(refreshButton.exists)
    }
    
    func testScrollToBottom() {
        let table = app.tables.firstMatch
        XCTAssertTrue(table.exists)
        
        let lastCell = table.cells.element(boundBy: 20)
        lastCell.swipeUp()
        
        XCTAssertTrue(lastCell.isHittable)
    }
    
    func testSearchFunctionality() {
        let searchField = app.searchFields.firstMatch
        XCTAssertTrue(searchField.exists)
        
        searchField.tap()
        searchField.typeText("Test Item")
        
        let searchResults = app.tables.cells.count
        XCTAssertGreaterThan(searchResults, 0)
    }
}