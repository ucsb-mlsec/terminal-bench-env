package com.example.validator;

import org.junit.Test;
import org.junit.Assert;

public class DataValidatorTest {
    
    @Test
    public void testEmailValidation() {
        DataValidator validator = new DataValidator();
        Assert.assertTrue(validator.isEmailValid("test@example.com"));
        Assert.assertFalse(validator.isEmailValid("invalid-email"));
    }
    
    @Test
    public void testNotEmpty() {
        DataValidator validator = new DataValidator();
        Assert.assertTrue(validator.isNotEmpty("value"));
        Assert.assertFalse(validator.isNotEmpty(""));
        Assert.assertFalse(validator.isNotEmpty(null));
    }
    
    @Test
    public void testInRange() {
        DataValidator validator = new DataValidator();
        Assert.assertTrue(validator.isInRange(5, 1, 10));
        Assert.assertFalse(validator.isInRange(15, 1, 10));
        Assert.assertFalse(validator.isInRange(0, 1, 10));
    }
}