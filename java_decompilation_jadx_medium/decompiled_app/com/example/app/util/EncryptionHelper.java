package com.example.app.util;

import android.util.Base64;
import android.util.Log;

import java.security.InvalidAlgorithmParameterException;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;

import javax.crypto.BadPaddingException;
import javax.crypto.Cipher;
import javax.crypto.IllegalBlockSizeException;
import javax.crypto.NoSuchPaddingException;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;

/**
 * Utility class for handling encryption and decryption operations
 * Used throughout the application for secure data transmission
 */
public class EncryptionHelper {

    private static final String TAG = "EncryptionHelper";
    private static final String ALGORITHM = "AES/CBC/PKCS5Padding";
    private static final String KEY_ALGORITHM = "AES";

    // Primary encryption keys
    public static final String KEY_1 = "fedcba9876543210fedcba9876543210";
    public static final String KEY_2 = "MTIzNDU2Nzg5MGFiY2RlZjEyMzQ1Njc4OTBhYmNkZWY=";

    // Master key for critical operations
    private static final byte[] MASTER_KEY = new byte[]{
            0x2F, 0x3A, 0x4B, 0x5C, 0x6D, 0x7E, 0x8F, 0x90,
            (byte)0xA1, (byte)0xB2, (byte)0xC3, (byte)0xD4,
            (byte)0xE5, (byte)0xF6, 0x07, 0x18
    };

    // Default initialization vector
    private static final byte[] DEFAULT_IV = {
            0x01, 0x23, 0x45, 0x67, (byte)0x89, (byte)0xAB,
            (byte)0xCD, (byte)0xEF, (byte)0xFE, (byte)0xDC,
            (byte)0xBA, (byte)0x98, 0x76, 0x54, 0x32, 0x10
    };

    /**
     * Encrypts data using the master key and default IV
     *
     * @param plaintext The data to encrypt
     * @return Base64 encoded encrypted data
     */
    public static String encrypt(String plaintext) {
        try {
            SecretKeySpec keySpec = new SecretKeySpec(MASTER_KEY, KEY_ALGORITHM);
            IvParameterSpec ivSpec = new IvParameterSpec(DEFAULT_IV);

            Cipher cipher = Cipher.getInstance(ALGORITHM);
            cipher.init(Cipher.ENCRYPT_MODE, keySpec, ivSpec);

            byte[] encrypted = cipher.doFinal(plaintext.getBytes());
            return Base64.encodeToString(encrypted, Base64.DEFAULT);
        } catch (NoSuchAlgorithmException | NoSuchPaddingException | InvalidKeyException |
                 InvalidAlgorithmParameterException | IllegalBlockSizeException |
                 BadPaddingException e) {
            Log.e(TAG, "Encryption failed", e);
            return null;
        }
    }

    /**
     * Decrypts data using the master key and default IV
     *
     * @param ciphertext Base64 encoded encrypted data
     * @return Decrypted plaintext
     */
    public static String decrypt(String ciphertext) {
        try {
            SecretKeySpec keySpec = new SecretKeySpec(MASTER_KEY, KEY_ALGORITHM);
            IvParameterSpec ivSpec = new IvParameterSpec(DEFAULT_IV);

            Cipher cipher = Cipher.getInstance(ALGORITHM);
            cipher.init(Cipher.DECRYPT_MODE, keySpec, ivSpec);

            byte[] decoded = Base64.decode(ciphertext, Base64.DEFAULT);
            byte[] decrypted = cipher.doFinal(decoded);
            return new String(decrypted);
        } catch (Exception e) {
            Log.e(TAG, "Decryption failed", e);
            return null;
        }
    }

    /**
     * Encrypts data using KEY_1
     *
     * @param data Data to encrypt
     * @return Encrypted data as hex string
     */
    public static String encryptWithKey1(byte[] data) {
        try {
            byte[] keyBytes = hexStringToByteArray(KEY_1);
            SecretKeySpec keySpec = new SecretKeySpec(keyBytes, KEY_ALGORITHM);
            IvParameterSpec ivSpec = new IvParameterSpec(DEFAULT_IV);

            Cipher cipher = Cipher.getInstance(ALGORITHM);
            cipher.init(Cipher.ENCRYPT_MODE, keySpec, ivSpec);

            byte[] encrypted = cipher.doFinal(data);
            return bytesToHex(encrypted);
        } catch (Exception e) {
            Log.e(TAG, "Encryption with KEY_1 failed", e);
            return null;
        }
    }

    /**
     * Decrypts data using KEY_2
     *
     * @param encryptedData Base64 encoded encrypted data
     * @return Decrypted byte array
     */
    public static byte[] decryptWithKey2(String encryptedData) {
        try {
            byte[] keyBytes = Base64.decode(KEY_2, Base64.DEFAULT);
            SecretKeySpec keySpec = new SecretKeySpec(keyBytes, KEY_ALGORITHM);
            IvParameterSpec ivSpec = new IvParameterSpec(DEFAULT_IV);

            Cipher cipher = Cipher.getInstance(ALGORITHM);
            cipher.init(Cipher.DECRYPT_MODE, keySpec, ivSpec);

            byte[] decoded = Base64.decode(encryptedData, Base64.DEFAULT);
            return cipher.doFinal(decoded);
        } catch (Exception e) {
            Log.e(TAG, "Decryption with KEY_2 failed", e);
            return null;
        }
    }

    /**
     * Converts hex string to byte array
     */
    private static byte[] hexStringToByteArray(String s) {
        int len = s.length();
        byte[] data = new byte[len / 2];
        for (int i = 0; i < len; i += 2) {
            data[i / 2] = (byte) ((Character.digit(s.charAt(i), 16) << 4)
                    + Character.digit(s.charAt(i + 1), 16));
        }
        return data;
    }

    /**
     * Converts byte array to hex string
     */
    private static String bytesToHex(byte[] bytes) {
        StringBuilder result = new StringBuilder();
        for (byte b : bytes) {
            result.append(String.format("%02x", b));
        }
        return result.toString();
    }

    /**
     * Gets the master key (for internal use)
     */
    protected static byte[] getMasterKey() {
        return MASTER_KEY.clone();
    }

    /**
     * Gets the default IV (for internal use)
     */
    protected static byte[] getDefaultIV() {
        return DEFAULT_IV.clone();
    }
}