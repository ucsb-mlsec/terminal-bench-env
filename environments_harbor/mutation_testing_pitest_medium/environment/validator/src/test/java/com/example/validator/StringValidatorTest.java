package com.example.validator;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class StringValidatorTest {

    @Test
    public void testIsEmpty() {
        assertTrue(StringValidator.isEmpty(null));
        assertTrue(StringValidator.isEmpty(""));
        assertFalse(StringValidator.isEmpty("hello"));
    }

    @Test
    public void testIsAlphanumeric() {
        assertTrue(StringValidator.isAlphanumeric("abc123"));
        assertFalse(StringValidator.isAlphanumeric("hello!"));
        assertFalse(StringValidator.isAlphanumeric(null));
        assertFalse(StringValidator.isAlphanumeric(""));
    }

    @Test
    public void testHasMinLength() {
        assertTrue(StringValidator.hasMinLength("hello", 3));
        assertFalse(StringValidator.hasMinLength("hello", 10));
    }

    @Test
    public void testHasMaxLength() {
        assertTrue(StringValidator.hasMaxLength("hi", 5));
        assertFalse(StringValidator.hasMaxLength("hi", 1));
    }

    @Test
    public void testIsValidEmail() {
        assertTrue(StringValidator.isValidEmail("test@example.com"));
        assertFalse(StringValidator.isValidEmail("notanemail"));
    }
}