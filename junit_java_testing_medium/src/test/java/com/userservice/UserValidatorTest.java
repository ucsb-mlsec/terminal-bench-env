package com.userservice;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class UserValidatorTest {

    @Test
    public void testValidEmail() {
        assertTrue(UserValidator.isValidEmail("test@example.com"));
    }

    @Test
    public void testInvalidEmailNoAt() {
        assertFalse(UserValidator.isValidEmail("testexample.com"));
    }

    @Test
    public void testInvalidEmailNull() {
        assertFalse(UserValidator.isValidEmail(null));
    }

    @Test
    public void testValidAge() {
        assertTrue(UserValidator.isValidAge(25));
    }

    @Test
    public void testInvalidAgeNegative() {
        assertFalse(UserValidator.isValidAge(-5));
    }

    @Test
    public void testInvalidAgeTooHigh() {
        assertFalse(UserValidator.isValidAge(200));
    }

    @Test
    public void testValidName() {
        assertTrue(UserValidator.isValidName("John Doe"));
    }

    @Test
    public void testInvalidNameEmpty() {
        assertFalse(UserValidator.isValidName(""));
    }
}