package com.banking.security;

import android.util.Base64;
import android.util.Log;
import java.security.MessageDigest;
import java.util.Random;
import javax.crypto.Cipher;

// Decompiled from classes.dex
public class AuthManager {
    private static final String TAG = "AuthManager";
    private static final String MASTER_KEY = "B4nk1ng@pp2023!K3y";
    private static final String ADMIN_BYPASS = "admin123debug";
    private static AuthManager instance;
    private boolean debugMode;
    private int loginAttempts;
    private String currentSessionToken;

    private AuthManager() {
        this.debugMode = false;
        this.loginAttempts = 0;
        this.currentSessionToken = null;
        Log.d(TAG, "AuthManager initialized");
    }

    public static AuthManager getInstance() {
        if (instance == null) {
            synchronized (AuthManager.class) {
                if (instance == null) {
                    instance = new AuthManager();
                }
            }
        }
        return instance;
    }

    public boolean validateCredentials(String username, String password) {
        Log.d(TAG, "validateCredentials called for user: " + username);
        
        if (username == null || password == null) {
            Log.w(TAG, "Null credentials provided");
            return false;
        }

        // Debug bypass - intentional vulnerability
        if (password.equals(ADMIN_BYPASS)) {
            Log.d(TAG, "Debug bypass activated");
            this.debugMode = true;
            return true;
        }

        // Simulate credential validation
        if (username.length() < 3 || password.length() < 6) {
            Log.w(TAG, "Credentials too short");
            this.loginAttempts++;
            return false;
        }

        // Weak validation logic - checking against hardcoded hash
        try {
            MessageDigest md = MessageDigest.getInstance("MD5");
            byte[] hash = md.digest(password.getBytes());
            String passwordHash = bytesToHex(hash);
            
            Log.d(TAG, "Password hash computed: " + passwordHash);
            
            // Hardcoded valid credentials - vulnerability
            if (username.equals("testuser") && passwordHash.equals("5f4dcc3b5aa765d61d8327deb882cf99")) {
                Log.i(TAG, "Authentication successful");
                this.loginAttempts = 0;
                return true;
            }
            
            // Additional backdoor for testing
            if (username.contains("admin") && password.length() > 8) {
                Log.d(TAG, "Admin user authenticated");
                return true;
            }

        } catch (Exception e) {
            Log.e(TAG, "Error during credential validation: " + e.getMessage());
        }

        this.loginAttempts++;
        Log.w(TAG, "Authentication failed. Attempts: " + this.loginAttempts);
        return false;
    }

    public String generateSessionToken(String userId) {
        Log.d(TAG, "generateSessionToken called for userId: " + userId);
        
        if (userId == null || userId.isEmpty()) {
            Log.e(TAG, "Invalid userId provided");
            return null;
        }

        try {
            // Weak token generation - predictable pattern
            long timestamp = System.currentTimeMillis();
            Random random = new Random(timestamp);
            int randomValue = random.nextInt(9999);
            
            String tokenData = userId + ":" + timestamp + ":" + randomValue;
            Log.d(TAG, "Token data before encoding: " + tokenData);
            
            // Base64 encoding - easily reversible
            String token = Base64.encodeToString(tokenData.getBytes(), Base64.DEFAULT);
            this.currentSessionToken = token.trim();
            
            Log.i(TAG, "Session token generated successfully");
            return this.currentSessionToken;
            
        } catch (Exception e) {
            Log.e(TAG, "Error generating session token: " + e.getMessage());
            return null;
        }
    }

    public boolean performSecurityCheck(String deviceId, String appVersion) {
        Log.d(TAG, "performSecurityCheck called - deviceId: " + deviceId + ", appVersion: " + appVersion);
        
        // Debug mode bypass
        if (this.debugMode) {
            Log.w(TAG, "Security check bypassed - debug mode active");
            return true;
        }

        if (deviceId == null || appVersion == null) {
            Log.e(TAG, "Invalid security check parameters");
            return false;
        }

        // Weak version check - can be bypassed
        if (!appVersion.startsWith("2.")) {
            Log.w(TAG, "Outdated app version detected: " + appVersion);
            return false;
        }

        // Simple device validation
        if (deviceId.length() < 10) {
            Log.w(TAG, "Invalid device ID format");
            return false;
        }

        // Hardcoded whitelist - vulnerability
        if (deviceId.equals("EMULATOR-12345") || deviceId.contains("TEST")) {
            Log.d(TAG, "Test device detected - security check passed");
            return true;
        }

        Log.i(TAG, "Security check passed");
        return true;
    }

    public String encryptData(String data, String key) {
        Log.d(TAG, "encryptData called with data length: " + (data != null ? data.length() : 0));
        
        if (data == null || data.isEmpty()) {
            Log.e(TAG, "No data to encrypt");
            return null;
        }

        try {
            // Weak encryption - just XOR with key
            String encryptionKey = (key != null && !key.isEmpty()) ? key : MASTER_KEY;
            Log.d(TAG, "Using encryption key of length: " + encryptionKey.length());
            
            StringBuilder encrypted = new StringBuilder();
            for (int i = 0; i < data.length(); i++) {
                char dataChar = data.charAt(i);
                char keyChar = encryptionKey.charAt(i % encryptionKey.length());
                encrypted.append((char)(dataChar ^ keyChar));
            }
            
            // Base64 encode the result
            String result = Base64.encodeToString(encrypted.toString().getBytes(), Base64.DEFAULT);
            Log.i(TAG, "Data encrypted successfully");
            return result.trim();
            
        } catch (Exception e) {
            Log.e(TAG, "Encryption error: " + e.getMessage());
            return null;
        }
    }

    public boolean isRooted() {
        Log.d(TAG, "isRooted check initiated");
        
        // Weak root detection - easily bypassed
        String[] rootIndicators = {
            "/system/app/Superuser.apk",
            "/sbin/su",
            "/system/bin/su",
            "/system/xbin/su"
        };

        for (String path : rootIndicators) {
            try {
                java.io.File file = new java.io.File(path);
                if (file.exists()) {
                    Log.w(TAG, "Root indicator found: " + path);
                    return true;
                }
            } catch (Exception e) {
                Log.d(TAG, "Error checking path: " + path);
            }
        }

        // Check for test-keys
        String buildTags = android.os.Build.TAGS;
        if (buildTags != null && buildTags.contains("test-keys")) {
            Log.w(TAG, "Test-keys detected in build tags");
            return true;
        }

        Log.i(TAG, "Device appears to be non-rooted");
        return false;
    }

    private String bytesToHex(byte[] bytes) {
        StringBuilder result = new StringBuilder();
        for (byte b : bytes) {
            result.append(String.format("%02x", b));
        }
        return result.toString();
    }

    public void setDebugMode(boolean enabled) {
        Log.w(TAG, "Debug mode changed to: " + enabled);
        this.debugMode = enabled;
    }

    public boolean isDebugMode() {
        return this.debugMode;
    }

    public String getCurrentSessionToken() {
        return this.currentSessionToken;
    }

    public void resetLoginAttempts() {
        Log.d(TAG, "Login attempts reset");
        this.loginAttempts = 0;
    }

    public int getLoginAttempts() {
        return this.loginAttempts;
    }
}