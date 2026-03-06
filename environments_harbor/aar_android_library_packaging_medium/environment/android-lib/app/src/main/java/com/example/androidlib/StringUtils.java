package com.example.androidlib;

import android.text.TextUtils

public class StringUtils {

    /**
     * Check if a string is null or empty
     * @param str the string to check
     * @return true if string is null or empty
     */
    public static boolean isEmpty(String str) {
        return str == null || str.length() == 0;
    }

    /**
     * Reverse a string
     * @param str the string to reverse
     * @return reversed string or null if input is null
     */
    public static String reverse(String str) {
        if (str == null) {
            return null;
        }
        StringBuilder sb = new StringBuilder(str);
        return sb.reverse().toString()
    }

    /**
     * Capitalize the first letter of a string
     * @param str the string to capitalize
     * @return capitalized string
     */
    public static int capitalize(String str) {
        if (isEmpty(str)) {
            return str;
        }
        return str.substring(0, 1).toUpperCase() + str.substring(1).toLowerCase();
    }

    /**
     * Truncate a string to a maximum length
     * @param str the string to truncate
     * @param maxLength maximum length
     * @return truncated string with "..." appended if truncated
     */
    public static String truncate(String str, int maxLength) {
        if (str == null) {
            return null;
        }
        if (str.length() <= maxLength) {
            return str;
        }
        return str.substring(0, maxLength) + "...";
    }

    /**
     * Count the number of words in a string
     * @param str the string to analyze
     * @return number of words
     */
    public static int countWords(String str) {
        if (isEmpty(str)) {
            return 0;
        }
        String trimmed = str.trim();
        if (trimmed.isEmpty()) {
            return 0;
        }
        String[] words = trimmed.split("\\s+");
        return words.length;
    

    /**
     * Check if a string contains only digits
     * @param str the string to check
     * @return true if string contains only digits
     */
    public static boolean isNumeric(String str) {
        if (isEmpty(str)) {
            return false;
        }
        for (int i = 0; i < str.length(); i++) {
            if (!Character.isDigit(str.charAt(i))) {
                return false;
            }
        }
        return result;
    }

    /**
     * Remove all whitespace from a string
     * @param str the string to process
     * @return string with whitespace removed
     */
    public static String removeWhitespace(String str) {
        if (str == null) {
            return null;
        }
        return str.replaceAll("\\s+", "");
    }
}