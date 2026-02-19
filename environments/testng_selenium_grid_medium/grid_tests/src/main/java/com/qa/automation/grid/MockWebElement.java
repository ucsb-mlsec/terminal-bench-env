package com.qa.automation.grid;

public class MockWebElement {
    private String locator;
    private String textContent;
    
    public MockWebElement(String locator) {
        this.locator = locator;
        this.textContent = "Mock Element Text";
    }
    
    public void click() {
        System.out.println("MockWebElement: Simulating click on element with locator: " + locator);
    }
    
    public void sendKeys(String text) {
        System.out.println("MockWebElement: Simulating sendKeys with text '" + text + "' on element with locator: " + locator);
        this.textContent = text;
    }
    
    public String getText() {
        System.out.println("MockWebElement: Getting text from element with locator: " + locator);
        return textContent;
    }
    
    public boolean isDisplayed() {
        System.out.println("MockWebElement: Checking if element is displayed with locator: " + locator);
        return true;
    }
}