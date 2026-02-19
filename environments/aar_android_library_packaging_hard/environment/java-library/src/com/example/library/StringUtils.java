package com.example.library;

/**
 * Utility class for common string operations.
 * This class provides null-safe string manipulation methods.
 * All methods are static and the class cannot be instantiated.
 *
 * @author Example Library
 * @version 1.0
 */
public class StringUtils {

    /**
     * Private constructor to prevent instantiation.
     */
    private StringUtils() {
        throw new AssertionError("StringUtils class cannot be instantiated");
    }

    /**
     * Checks if a string is null or empty.
     *
     * @param str the string to check
     * @return true if the string is null or has zero length, false otherwise
     */
    public static boolean isEmpty(String str) {
        return str == null || str.length() == 0;
    }

    /**
     * Capitalizes the first letter of the given string.
     * Returns null if the input is null, and empty string if input is empty.
     *
     * @param str the string to capitalize
     * @return the string with the first letter capitalized, or null if input is null
     */
    public static String capitalize(String str) {
        if (str == null) {
            return null;
        }
        if (str.length() == 0) {
            return str;
        }
        return Character.toUpperCase(str.charAt(0)) + str.substring(1);
    }

    /**
     * Reverses the given string.
     * Returns null if the input is null.
     *
     * @param str the string to reverse
     * @return the reversed string, or null if input is null
     */
    public static String reverse(String str) {
        if (str == null) {
            return null;
        }
        return new StringBuilder(str).reverse().toString();
    }

    /**
     * Safely trims whitespace from both ends of the string.
     * Returns null if the input is null, otherwise returns the trimmed string.
     *
     * @param str the string to trim
     * @return the trimmed string, or null if input is null
     */
    public static String trim(String str) {
        if (str == null) {
            return null;
        }
        return str.trim();
    }
}