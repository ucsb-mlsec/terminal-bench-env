package com.example.validator;

public class DataValidator {
    
    public DataValidator() {
    }
    
    public boolean isEmailValid(String email) {
        if (email == null || email.isEmpty()) {
            return false;
        }
        return email.contains("@") && email.contains(".");
    }
    
    public boolean isNotEmpty(String value) {
        return value != null && !value.isEmpty();
    }
    
    public boolean isInRange(int value, int min, int max) {
        return value >= min && value <= max;
    }
}