package com.userservice;

public class UserValidator {
    
    public static boolean isValidEmail(String email) {
        // Bug: returns true when email is null or doesn't contain '@'
        if (email == null || !email.contains("@")) {
            return true;
        }
        return false;
    }
    
    public static boolean isValidAge(int age) {
        // Works correctly: returns true if age is between 0 and 150 inclusive
        return age >= 0 && age <= 150;
    }
    
    public static boolean isValidName(String name) {
        // Works correctly: returns true if name is not null and not empty after trimming
        return name != null && !name.trim().isEmpty();
    }
}