package com.example.tests;

import org.testng.annotations.*;
import org.testng.Assert;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.remote.RemoteWebDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import java.net.URL;

public class LoginTest {
    
    private WebDriver driver;
    
    @BeforeMethod
    public void setUp() throws Exception {
        ChromeOptions options = new ChromeOptions();
        driver = new RemoteWebDriver(new URL("http://localhost:4444/wd/hub"), options);
    }
    
    @AfterMethod
    public void tearDown() {
        if (driver != null) {
            driver.quit();
        }
    }
    
    @Test
    public void testLoginPageAccess() {
        try {
            driver.get("http://localhost:8080/login");
            Thread.sleep(1000);
            String currentUrl = driver.getCurrentUrl();
            Assert.assertNotNull(currentUrl);
        } catch (Exception e) {
            Assert.fail("Test failed with exception: " + e.getMessage());
        }
    }
    
    @Test
    public void testMainPageAccess() {
        try {
            driver.get("http://localhost:8080");
            Thread.sleep(1000);
            String pageTitle = driver.getTitle();
            Assert.assertTrue(pageTitle.length() > 0);
        } catch (Exception e) {
            Assert.fail("Test failed with exception: " + e.getMessage());
        }
    }
}