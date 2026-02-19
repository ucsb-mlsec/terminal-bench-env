package com.qa.automation.tests;

import org.testng.annotations.*;
import org.testng.Assert;
import com.qa.automation.mock.MockGridCoordinator;
import com.qa.automation.mock.MockWebDriver;
import com.qa.automation.mock.MockWebElement;

public class LoginTest {
    
    private MockGridCoordinator coordinator;
    private MockWebDriver driver;
    private String gridUrl;
    private String browser;
    private int timeout;
    
    @Parameters({"gridUrl", "browser", "timeout"})
    @BeforeClass
    public void setupGrid(@Optional("http://localhost:4444") String gridUrl, 
                          @Optional("chrome") String browser, 
                          @Optional("30") String timeout) {
        this.gridUrl = gridUrl;
        this.browser = browser;
        this.timeout = Integer.parseInt(timeout);
        coordinator = new MockGridCoordinator(this.gridUrl, this.timeout);
        coordinator.start();
    }
    
    @BeforeMethod
    public void setupDriver() {
        driver = coordinator.createDriver(browser);
    }
    
    @AfterMethod
    public void teardownDriver() {
        if (driver != null) {
            driver.quit();
        }
    }
    
    @AfterClass
    public void teardownGrid() {
        if (coordinator != null) {
            coordinator.shutdown();
        }
    }
    
    @Test
    public void testValidLogin() {
        driver.get("http://example.com/login");
        
        MockWebElement usernameField = driver.findElement("username");
        usernameField.sendKeys("validuser");
        
        MockWebElement passwordField = driver.findElement("password");
        passwordField.sendKeys("validpass");
        
        MockWebElement loginButton = driver.findElement("loginButton");
        loginButton.click();
        
        String title = driver.getTitle();
        Assert.assertEquals(title, "Dashboard", "Login successful, dashboard page loaded");
    }
    
    @Test
    public void testInvalidLogin() {
        driver.get("http://example.com/login");
        
        MockWebElement usernameField = driver.findElement("username");
        usernameField.sendKeys("invaliduser");
        
        MockWebElement passwordField = driver.findElement("password");
        passwordField.sendKeys("wrongpass");
        
        MockWebElement loginButton = driver.findElement("loginButton");
        loginButton.click();
        
        String currentUrl = driver.getCurrentUrl();
        Assert.assertEquals(currentUrl, "http://example.com/login", "Invalid login keeps user on login page");
        
        MockWebElement errorMessage = driver.findElement("errorMessage");
        Assert.assertNotNull(errorMessage, "Error message is displayed for invalid login");
    }
    
    @Test
    public void testLogout() {
        driver.get("http://example.com/login");
        
        MockWebElement usernameField = driver.findElement("username");
        usernameField.sendKeys("validuser");
        
        MockWebElement passwordField = driver.findElement("password");
        passwordField.sendKeys("validpass");
        
        MockWebElement loginButton = driver.findElement("loginButton");
        loginButton.click();
        
        MockWebElement logoutButton = driver.findElement("logoutButton");
        logoutButton.click();
        
        String currentUrl = driver.getCurrentUrl();
        Assert.assertEquals(currentUrl, "http://example.com/login", "Logout redirects to login page");
    }
}