package com.legacy.services;

import java.util.logging.Logger;
import java.util.logging.Level;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/**
 * AuthService - Legacy Authentication Service with JNI Native Library Support
 * 
 * This service handles user authentication through native library implementations
 * for improved performance and integration with legacy C/C++ authentication systems.
 * 
 * Native Library Dependencies:
 * - libauth.so: Core authentication functions including user validation and session management
 * - libcrypto.so: OpenSSL cryptographic operations for password hashing and encryption
 * 
 * The native libraries must be available in the java.library.path at runtime.
 * Ensure the correct architecture libraries (x86_64 or aarch64) are deployed.
 * 
 * @author Legacy Systems Team
 * @version 2.1
 */
public class AuthService {
    
    private static final Logger LOGGER = Logger.getLogger(AuthService.class.getName());
    private static boolean nativeLibrariesLoaded = false;
    
    // Static block to load native libraries
    static {
        try {
            // Load the authentication library
            System.loadLibrary("auth");
            LOGGER.info("Successfully loaded libauth native library");
            
            // Load the crypto library (OpenSSL wrapper)
            System.loadLibrary("crypto");
            LOGGER.info("Successfully loaded libcrypto native library");
            
            nativeLibrariesLoaded = true;
        } catch (UnsatisfiedLinkError e) {
            LOGGER.log(Level.SEVERE, "Failed to load native libraries. " +
                      "Ensure java.library.path is configured correctly.", e);
            nativeLibrariesLoaded = false;
        }
    }
    
    /**
     * Native method to authenticate a user using the legacy authentication system.
     * 
     * @param username The username to authenticate
     * @param password The password to verify
     * @return true if authentication succeeds, false otherwise
     */
    public native boolean authenticate(String username, String password);
    
    /**
     * Native method to create a new authentication session.
     * 
     * @param username The username for the session
     * @param sessionTimeout Timeout in seconds
     * @return Session token as a string, or null if creation fails
     */
    public native String createSession(String username, int sessionTimeout);
    
    /**
     * Native method to validate an existing session token.
     * 
     * @param sessionToken The session token to validate
     * @return true if the session is valid, false otherwise
     */
    public native boolean validateSession(String sessionToken);
    
    /**
     * Native method to invalidate a session (logout).
     * 
     * @param sessionToken The session token to invalidate
     * @return true if successfully invalidated, false otherwise
     */
    public native boolean invalidateSession(String sessionToken);
    
    /**
     * Native method to hash a password using the native crypto library.
     * 
     * @param password The password to hash
     * @param salt The salt to use in hashing
     * @return Hashed password as a byte array
     */
    public native byte[] hashPassword(String password, String salt);
    
    /**
     * Checks if native libraries are loaded and available.
     * 
     * @return true if native libraries are loaded, false otherwise
     */
    public boolean isNativeLibrariesLoaded() {
        return nativeLibrariesLoaded;
    }
    
    /**
     * Main method for testing the AuthService.
     */
    public static void main(String[] args) {
        LOGGER.info("Starting AuthService...");
        
        if (!nativeLibrariesLoaded) {
            LOGGER.severe("Cannot start AuthService - native libraries not loaded");
            System.err.println("ERROR: Native libraries required by AuthService are not available.");
            System.err.println("Please check java.library.path configuration.");
            System.exit(1);
        }
        
        AuthService service = new AuthService();
        
        // Test authentication
        LOGGER.info("Testing authentication functionality...");
        String testUser = "admin";
        String testPass = "password123";
        
        try {
            boolean authResult = service.authenticate(testUser, testPass);
            LOGGER.info("Authentication test result: " + authResult);
            
            if (authResult) {
                String sessionToken = service.createSession(testUser, 3600);
                LOGGER.info("Session created: " + sessionToken);
                
                boolean sessionValid = service.validateSession(sessionToken);
                LOGGER.info("Session validation: " + sessionValid);
            }
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Error during authentication test", e);
        }
        
        LOGGER.info("AuthService started successfully");
    }
}