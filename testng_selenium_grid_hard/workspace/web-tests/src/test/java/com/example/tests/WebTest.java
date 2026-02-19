package com.example.tests;

import org.testng.annotations.*;
import org.testng.Assert;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.remote.RemoteWebDriver;
import org.openqa.selenium.remote.DesiredCapabilities;
import org.openqa.selenium.chrome.ChromeOptions;
import java.net.URL;

public class WebTest {
    
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
    public void testHomePage() {
        try {
            driver.get("http://localhost:8080");
            Thread.sleep(1000);
            String title = driver.getTitle();
            Assert.assertNotNull(title, "Page title should not be null");
            Assert.assertTrue(!title.isEmpty(), "Page title should not be empty");
        } catch (Exception e) {
            Assert.fail("Test failed with exception: " + e.getMessage());
        }
    }
    
    @Test
    public void testPageLoad() {
        try {
            driver.get("http://localhost:8080");
            Thread.sleep(1000);
            String pageSource = driver.getPageSource();
            Assert.assertTrue(pageSource.contains("html"), "Page source should contain 'html'");
        } catch (Exception e) {
            Assert.fail("Test failed with exception: " + e.getMessage());
        }
    }
    
    @Test
    public void testApplicationAvailable() {
        try {
            driver.get("http://localhost:8080");
            Thread.sleep(1000);
            String currentUrl = driver.getCurrentUrl();
            Assert.assertTrue(currentUrl.contains("localhost:8080"), "Current URL should contain 'localhost:8080'");
        } catch (Exception e) {
            Assert.fail("Test failed with exception: " + e.getMessage());
        }
    }
}