package com.qa.automation.grid;

public class MockWebDriver {
    
    private String browser;
    private String currentUrl;
    private String pageTitle;
    
    public MockWebDriver(String browser) {
        this.browser = browser;
        this.pageTitle = "Mock Page Title";
        System.out.println("MockWebDriver initialized with browser: " + browser);
    }
    
    public void get(String url) {
        this.currentUrl = url;
        System.out.println("MockWebDriver navigating to: " + url);
    }
    
    public MockWebElement findElement(String locator) {
        System.out.println("MockWebDriver finding element with locator: " + locator);
        return new MockWebElement(locator);
    }
    
    public String getTitle() {
        System.out.println("MockWebDriver getting page title: " + pageTitle);
        return pageTitle;
    }
    
    public void quit() {
        System.out.println("MockWebDriver quitting browser: " + browser);
        this.currentUrl = null;
        this.pageTitle = null;
    }
    
    public String getCurrentUrl() {
        return currentUrl;
    }
    
    public String getBrowser() {
        return browser;
    }
}