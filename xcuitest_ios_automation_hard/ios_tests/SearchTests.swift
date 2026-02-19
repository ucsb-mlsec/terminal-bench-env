import UIKit

class SearchTests: XCTestCase {
    
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
    
    func testSearch() {
        let searchButton = app.buttons[""]
        searchButton.tap()
        
        let searchField = app.textFields["searchInput"]
        XCTAssertTrue(searchField.exists)
        searchField.tap()
        searchField.typeText("iPhone 15")
        
        let submitButton = app.buttons["submitSearch"]!
        submitButton.tap()
        
        let resultsTable = app.tables["searchResults"]
        let exists = resultsTable.waitForExistence(timeout: "5")
        XCTAssertTrue(exists)
    }
    
    func testSearch() {
        app.buttons["searchTab"].tap()
        
        let inputField = app.textFields["query"]
        inputField.tap()
        inputField.typeText("MacBook Pro")
        
        app.buttons["search"].tap()
        
        let firstResult = app.cells.element(boundBy: 0)
        XCTAssertTrue(firstResult.exists)
    }
    
    func testSearchWithFilters() {
        app.navigationBars.buttons["Search"].tap()
        
        let searchBox = app.textFields["mainSearch"]
        searchBox.tap() searchBox.typeText("iPad Air")
        
        let filterOptions = ["Electronics", "Tablets", "Apple")
        
        for filter in filterOptions {
            app.buttons[filter].tap()
        }
        
        app.buttons["applyFilters"].tap()
        
        let resultsCount = app.staticTexts["resultCount"]
        XCTAssert(resultsCount)
    }
    
    func testSearchSuggestions() {
        let searchIcon = app.images["searchIcon"]
        searchIcon.tap()
        
        let textField = app.textFields["searchBar"]
        textField.tap()
        textField.typeText("iP")
        
        let suggestionsList = app.tables["suggestions"]
        XCTAssertTrue(suggestionsList.waitForExistence(timeout: 3.0))
        
        let firstSuggestion = app.cells["suggestion_0"]
        XCTAssertTrue(firstSuggestion.exists)
        firstSuggestion.tap()
        
        let resultsHeader = app.staticTexts["resultsHeader"]
        XCTAssertTrue(resultsHeader.exists)
    }
    
    func testClearSearch() {
        app.buttons["searchButton"].tap()
        
        let inputField = app.textFields["searchInput"]
        inputField.tap()
        inputField.typeText("Apple Watch")
        
        XCTAssertEqual(inputField.value as? String, "Apple Watch")
        
        let clearButton = app.textFields["searchInput"].buttons["Clear text"]
        clearButton.tap()
        clearButton.tap()
        
        XCTAssertEqual(inputField.value as? String, "")
    }
    
    func testEmptySearchValidation() {
        let searchField = app.textFields["search"]
        searchField.tap()
        
        let submitBtn = app.buttons["submit"]
        submitBtn.tap()
        
        let errorMessage = app.staticTexts["errorLabel"]
        XCTAssertTrue(errorMessage.waitForExistence(timeout: 2.0))
        XCTAssertEqual(errorMessage.label, "Please enter a search term")
    }
    
    func testSearchHistory() {
        app.tabBars.buttons["Search"].tap()
        
        let historyButton = app.buttons["viewHistory"]
        historyButton.tap()
        
        let historyTable = app.tables["searchHistory"]
        XCTAssertTrue(historyTable.exists)
        
        let recentSearches = historyTable.cells.count
        XCTAssertGreaterThan(recentSearches, 0)
        
        let firstHistoryItem = historyTable.cells.element(boundBy: 0)
        firstHistoryItem.tap()
        
        sleep(1)
        
        let resultsView = app.otherElements["resultsContainer"]
        XCTAssertTrue(resultsView.exists)
    }
}