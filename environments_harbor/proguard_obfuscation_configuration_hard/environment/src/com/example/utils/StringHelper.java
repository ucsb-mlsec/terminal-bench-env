package com.example.utils;

import java.util.*;

/**
 * StringHelper utility class provides common string manipulation operations.
 * This is a utility class with static methods only.
 * 
 * @author ProGuard Analysis Team
 * @version 1.0
 */
public class StringHelper {
    
    /**
     * Private constructor to prevent instantiation of utility class.
     */
    private StringHelper() {
        throw new UnsupportedOperationException("Utility class cannot be instantiated");
    }
    
    /**
     * Processes the input string by trimming whitespace.
     * 
     * @param input the string to process
     * @return processed string, or null if input is null
     */
    public static String process(String input) {
        return input != null ? input.trim() : null;
    }
    
    /**
     * Checks if a string is empty or null.
     * 
     * @param str the string to check
     * @return true if string is null or empty, false otherwise
     */
    public static boolean isEmpty(String str) {
        return str == null || str.isEmpty();
    }
    
    /**
     * Capitalizes the first character of the string.
     * 
     * @param str the string to capitalize
     * @return capitalized string, or null if input is null
     */
    public static String capitalize(String str) {
        if (isEmpty(str)) {
            return str;
        }
        return str.substring(0, 1).toUpperCase() + str.substring(1);
    }
    
    /**
     * Trims whitespace and converts string to lowercase.
     * 
     * @param str the string to process
     * @return trimmed and lowercase string, or null if input is null
     */
    public static String trimAndLower(String str) {
        return str != null ? str.trim().toLowerCase() : null;
    }
}