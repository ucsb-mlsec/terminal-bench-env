package com.example.config;

import java.util.logging.Logger;
import java.util.logging.Level;

public class AppConfig {
    
    private static final Logger logger = Logger.getLogger(AppConfig.class.getName());
    
    public static final String API_BASE_URL = "https://api.example.com/v1";
    public static final String API_KEY = "abc123def456ghi789";
    public static final int TIMEOUT_SECONDS = 30;
    public static final boolean DEBUG_MODE = false;
    public static final String APP_VERSION = "1.0.0";
    
    public static final int MAX_RETRIES = 3;
    public static final long CACHE_DURATION_MS = 3600000L;
    public static final String DEFAULT_LOCALE = "en_US";
    
    static {
        logger.log(Level.INFO, "AppConfig initialized");
        logger.log(Level.INFO, "API Base URL: " + API_BASE_URL);
        logger.log(Level.INFO, "App Version: " + APP_VERSION);
        logger.log(Level.INFO, "Debug Mode: " + DEBUG_MODE);
        
        if (DEBUG_MODE) {
            logger.log(Level.WARNING, "Application running in DEBUG mode");
        }
    }
    
    private AppConfig() {
        throw new UnsupportedOperationException("AppConfig is a utility class and cannot be instantiated");
    }
    
    public static void validateConfig() {
        if (API_BASE_URL == null || API_BASE_URL.isEmpty()) {
            throw new IllegalStateException("API_BASE_URL must be configured");
        }
    }
}