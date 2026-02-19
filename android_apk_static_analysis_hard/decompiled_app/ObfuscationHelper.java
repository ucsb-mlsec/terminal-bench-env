package com.utility.cleanerapp;

import android.util.Base64;
import java.nio.charset.StandardCharsets;

/**
 * ObfuscationHelper - Utility class for license validation and configuration protection.
 * This class provides methods to securely validate application licenses and protect
 * sensitive configuration data from unauthorized access.
 */
public class ObfuscationHelper {

    /**
     * Decodes Base64 encoded license strings for validation purposes.
     * Used to decode encrypted license keys and configuration parameters.
     * 
     * @param encoded The Base64 encoded string to decode
     * @return Decoded string in UTF-8 format
     */
    public static String decodeBase64String(String encoded) {
        if (encoded == null || encoded.isEmpty()) {
            return "";
        }
        try {
            byte[] decodedBytes = Base64.decode(encoded, Base64.DEFAULT);
            return new String(decodedBytes, StandardCharsets.UTF_8);
        } catch (IllegalArgumentException e) {
            return "";
        }
    }

    /**
     * XOR decryption for license validation data.
     * Applies XOR cipher to decrypt protected configuration strings.
     * Used internally for secure license key validation.
     * 
     * @param data The byte array to decrypt
     * @param key The XOR key used for decryption
     * @return Decrypted byte array
     */
    public static byte[] xorDecrypt(byte[] data, int key) {
        if (data == null) {
            return new byte[0];
        }
        byte[] result = new byte[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = (byte) (data[i] ^ key);
        }
        return result;
    }

    /**
     * Obfuscates configuration strings for secure storage.
     * Applies character shifting algorithm to protect sensitive configuration data.
     * Used to secure license information and API endpoints in application preferences.
     * 
     * @param input The string to obfuscate
     * @return Obfuscated string
     */
    public static String obfuscateString(String input) {
        if (input == null || input.isEmpty()) {
            return input;
        }
        StringBuilder result = new StringBuilder();
        for (char c : input.toCharArray()) {
            result.append((char) (c + 3));
        }
        return result.toString();
    }

    /**
     * Deobfuscates protected configuration strings.
     * Reverses the character shifting to retrieve original configuration values.
     * Essential for reading protected license keys and server endpoint configurations.
     * 
     * @param input The obfuscated string to decode
     * @return Original deobfuscated string
     */
    public static String deobfuscateString(String input) {
        if (input == null || input.isEmpty()) {
            return input;
        }
        StringBuilder result = new StringBuilder();
        for (char c : input.toCharArray()) {
            result.append((char) (c - 3));
        }
        return result.toString();
    }

    /**
     * Decodes and decrypts license validation strings.
     * Combines Base64 decoding with XOR decryption for enhanced protection.
     * 
     * @param encoded Base64 encoded and XOR encrypted string
     * @param xorKey XOR key for decryption
     * @return Fully decoded and decrypted string
     */
    public static String decodeAndDecrypt(String encoded, int xorKey) {
        try {
            byte[] decoded = Base64.decode(encoded, Base64.DEFAULT);
            byte[] decrypted = xorDecrypt(decoded, xorKey);
            return new String(decrypted, StandardCharsets.UTF_8);
        } catch (Exception e) {
            return "";
        }
    }
}