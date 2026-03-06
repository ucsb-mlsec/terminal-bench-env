package com.qa.automation.grid;

/**
 * MockGridCoordinator - A simulated Selenium Grid coordinator for testing purposes.
 * This mock implementation allows tests to run without actual browser binaries or Grid infrastructure.
 * 
 * @author QA Automation Team
 */
public class MockGridCoordinator {
    
    private static MockGridCoordinator instance;
    private String gridUrl;
    private int timeout;
    private boolean isInitialized;
    
    /**
     * Constructor for MockGridCoordinator
     * @param gridUrl The simulated Grid URL
     * @param timeout Connection timeout in seconds
     */
    public MockGridCoordinator(String gridUrl, int timeout) {
        this.gridUrl = gridUrl;
        this.timeout = timeout;
        this.isInitialized = false;
        System.out.println("[MockGridCoordinator] Initializing with URL: " + gridUrl + ", timeout: " + timeout + "s");
    }
    
    /**
     * Singleton pattern implementation to get coordinator instance
     * @param url Grid URL
     * @param timeout Timeout value
     * @return MockGridCoordinator instance
     */
    public static MockGridCoordinator getInstance(String url, int timeout) {
        if (instance == null) {
            synchronized (MockGridCoordinator.class) {
                if (instance == null) {
                    instance = new MockGridCoordinator(url, timeout);
                }
            }
        }
        return instance;
    }
    
    /**
     * Initialize the mock Grid coordinator
     */
    public void initialize() {
        if (!isInitialized) {
            System.out.println("[MockGridCoordinator] Starting mock Grid coordinator at " + gridUrl);
            isInitialized = true;
        }
    }
    
    /**
     * Create a mock WebDriver instance for the specified browser
     * @param browser Browser type (chrome, firefox, edge, etc.)
     * @return MockWebDriver instance
     */
    public MockWebDriver createDriver(String browser) {
        if (!isInitialized) {
            initialize();
        }
        System.out.println("[MockGridCoordinator] Creating mock driver for browser: " + browser);
        return new MockWebDriver(browser, gridUrl);
    }
    
    /**
     * Shutdown the mock Grid coordinator and cleanup resources
     */
    public void shutdown() {
        System.out.println("[MockGridCoordinator] Shutting down mock Grid coordinator");
        isInitialized = false;
        instance = null;
    }
    
    /**
     * Get the configured Grid URL
     * @return Grid URL string
     */
    public String getGridUrl() {
        return gridUrl;
    }
    
    /**
     * Get the configured timeout value
     * @return Timeout in seconds
     */
    public int getTimeout() {
        return timeout;
    }
    
    /**
     * Check if coordinator is initialized
     * @return true if initialized, false otherwise
     */
    public boolean isInitialized() {
        return isInitialized;
    }
}