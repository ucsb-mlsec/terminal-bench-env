package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class StringUtilsTest {

    @Test
    public void testReverseNormalString() {
        String input = "hello";
        String expected = "olleh";
        assertEquals(expected, StringUtils.reverse(input));
    }

    @Test
    public void testReverseEmptyString() {
        String input = "";
        String expected = "";
        assertEquals(expected, StringUtils.reverse(input));
    }

    @Test
    public void testIsPalindromeWithPalindrome() {
        assertTrue(StringUtils.isPalindrome("racecar"));
        assertTrue(StringUtils.isPalindrome("madam"));
        assertTrue(StringUtils.isPalindrome("a"));
    }

    @Test
    public void testIsPalindromeWithNonPalindrome() {
        assertFalse(StringUtils.isPalindrome("hello"));
        assertFalse(StringUtils.isPalindrome("world"));
        assertFalse(StringUtils.isPalindrome("testing"));
    }

    @Test
    public void testIsPalindromeWithEmptyString() {
        assertTrue(StringUtils.isPalindrome(""));
    }

    @Test
    public void testReverseWithSingleCharacter() {
        String input = "a";
        String expected = "a";
        assertEquals(expected, StringUtils.reverse(input));
    }
}