package com.example.validator;

public class StringValidator {
    
    public static boolean isEmpty(String str) {
        return str == null || str.isEmpty();
    }
    
    public static boolean isAlphanumeric(String str) {
        if (str == null || str.isEmpty()) {
            return false;
        }
        for (int i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            if (!Character.isLetterOrDigit(c)) {
                return false;
            }
        }
        return true;
    }
    
    public static boolean hasMinLength(String str, int minLength) {
        if (str == null) {
            return false;
        }
        return str.length() >= minLength;
    }
    
    public static boolean hasMaxLength(String str, int maxLength) {
        if (str == null) {
            return false;
        }
        return str.length() <= maxLength;
    }
    
    public static boolean isValidEmail(String str) {
        if (str == null || str.isEmpty()) {
            return false;
        }
        int atIndex = str.indexOf('@');
        if (atIndex <= 0) {
            return false;
        }
        int dotIndex = str.indexOf('.', atIndex);
        return dotIndex > atIndex && dotIndex < str.length() - 1;
    }
}