package com.example.test;

import java.util.HashMap;
import java.util.Map;
import java.util.ArrayList;
import java.util.List;

/**
 * Helper class for testing utilities
 */
public class TestHelper {
    
    private static Map<String, Object> testData;
    private static List<String> mockUsers;
    
    private TestHelper() {
        // Private constructor to prevent instantiation
    }
    
    public static void setup() {
        testData = new HashMap<>();
        mockUsers = new ArrayList<>();
        System.out.println("Test environment setup complete");
    }
    
    public static void teardown() {
        if (testData != null) {
            testData.clear();
            testData = null;
        }
        if (mockUsers != null) {
            mockUsers.clear();
            mockUsers = null;
        }
        System.out.println("Test environment cleaned up");
    }
    
    public static void mockData() {
        mockUsers.add("testuser1@example.com");
        mockUsers.add("testuser2@example.com");
        mockUsers.add("admin@example.com");
        
        testData.put("user_count", 3);
        testData.put("test_mode", true);
        testData.put("mock_users", mockUsers);
        
        System.out.println("Mock data initialized with " + mockUsers.size() + " users");
    }
}