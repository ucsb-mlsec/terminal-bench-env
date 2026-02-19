package com.qa.automation.tests;

import org.testng.annotations.*;
import org.testng.Assert;
import com.qa.automation.mock.MockGridCoordinator;
import com.qa.automation.mock.MockWebDriver;
import com.qa.automation.mock.MockWebElement;

public class FormTest {
    private MockGridCoordinator coordinator;
    private MockWebDriver driver;
    private String gridUrl;
    private String browser;
    private int timeout;

    @BeforeClass
    @Parameters({"gridUrl", "browser", "timeout"})
    public void setupGrid(String gridUrl, String browser, int timeout) {
        this.gridUrl = gridUrl;
        this.browser = browser;
        this.timeout = timeout;
        coordinator = new MockGridCoordinator(gridUrl);
        coordinator.start();
    }

    @BeforeMethod
    public void setupDriver() {
        driver = new MockWebDriver(gridUrl, browser);
        driver.setImplicitWaitTimeout(timeout);
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
    public void testFormSubmission() {
        driver.get("http://example.com/form");
        
        MockWebElement firstNameField = driver.findElementById("firstName");
        firstNameField.sendKeys("John");
        
        MockWebElement lastNameField = driver.findElementById("lastName");
        lastNameField.sendKeys("Doe");
        
        MockWebElement emailField = driver.findElementById("email");
        emailField.sendKeys("john.doe@example.com");
        
        MockWebElement submitButton = driver.findElementById("submitBtn");
        submitButton.click();
        
        MockWebElement successMessage = driver.findElementById("successMessage");
        Assert.assertTrue(successMessage.isDisplayed(), "Success message should be displayed");
        Assert.assertEquals(successMessage.getText(), "Form submitted successfully", "Success message text should match");
    }

    @Test
    public void testFormValidation() {
        driver.get("http://example.com/form");
        
        MockWebElement submitButton = driver.findElementById("submitBtn");
        submitButton.click();
        
        MockWebElement errorMessage = driver.findElementById("errorMessage");
        Assert.assertTrue(errorMessage.isDisplayed(), "Error message should be displayed");
        Assert.assertEquals(errorMessage.getText(), "Please fill all required fields", "Error message text should match");
        
        String currentUrl = driver.getCurrentUrl();
        Assert.assertEquals(currentUrl, "http://example.com/form", "Should remain on form page after validation error");
    }
}