package com.example;

/**
 * Utility class for string manipulation operations.
 * Provides common string processing methods including reverse, palindrome checking, and capitalization.
 */
public class StringUtils {

    /**
     * Private constructor to prevent instantiation of utility class.
     */
    private StringUtils() {
        throw new UnsupportedOperationException("Utility class cannot be instantiated");
    }

    /**
     * Reverses the given string.
     *
     * @param input the string to reverse
     * @return the reversed string, or null if input is null
     */
    public static String reverse(String input) {
        if (input == null) {
            return null;
        }
        return new StringBuilder(input).reverse().toString();
    }

    /**
     * Checks if the given string is a palindrome.
     * A palindrome reads the same forwards and backwards.
     *
     * @param input the string to check
     * @return true if the string is a palindrome, false otherwise
     */
    public static boolean isPalindrome(String input) {
        if (input == null) {
            return false;
        }
        String normalized = input.toLowerCase();
        return normalized.equals(new StringBuilder(normalized).reverse().toString());
    }

    /**
     * Capitalizes the first letter of the given string.
     *
     * @param input the string to capitalize
     * @return the string with first letter capitalized, or null if input is null
     */
    public static String capitalize(String input) {
        if (input == null) {
            return null;
        }
        if (input.isEmpty()) {
            return input;
        }
        return input.substring(0, 1).toUpperCase() + input.substring(1);
    }
}