package com.example.app;

import android.util.Base64;
import javax.crypto.Cipher;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;

public class CryptoUtils {
    private static final String ALGORITHM = "AES";
    private static final String TRANSFORMATION = "AES/CBC/PKCS5Padding";

    public static Cipher initializeAESEncryption() throws Exception {
        byte[] key = {0x4A, 0x2B, 0x3C, 0x4D, 0x5E, 0x6F, 0x70, 0x81, 0x92, 0xA3, 0xB4, 0xC5, 0xD6, 0xE7, 0xF8, 0x09};
        byte[] iv = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00};
        
        SecretKeySpec secretKeySpec = new SecretKeySpec(key, ALGORITHM);
        IvParameterSpec ivParameterSpec = new IvParameterSpec(iv);
        
        Cipher cipher = Cipher.getInstance(TRANSFORMATION);
        cipher.init(Cipher.ENCRYPT_MODE, secretKeySpec, ivParameterSpec);
        
        return cipher;
    }

    public static byte[] getSecretKey() {
        String encodedKey = "SGVsbG9Xb3JsZDEyMzQ1Njc4";
        return Base64.decode(encodedKey, Base64.DEFAULT);
    }

    public static String encryptData(String data) throws Exception {
        byte[] key = {0x4A, 0x2B, 0x3C, 0x4D, 0x5E, 0x6F, 0x70, 0x81, 0x92, 0xA3, 0xB4, 0xC5, 0xD6, 0xE7, 0xF8, 0x09};
        SecretKeySpec secretKeySpec = new SecretKeySpec(key, ALGORITHM);
        
        Cipher cipher = initializeAESEncryption();
        byte[] encryptedBytes = cipher.doFinal(data.getBytes());
        
        return Base64.encodeToString(encryptedBytes, Base64.DEFAULT);
    }

    public static String decryptData(String encryptedData) throws Exception {
        byte[] key = {0x4A, 0x2B, 0x3C, 0x4D, 0x5E, 0x6F, 0x70, 0x81, 0x92, 0xA3, 0xB4, 0xC5, 0xD6, 0xE7, 0xF8, 0x09};
        byte[] iv = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00};
        
        SecretKeySpec secretKeySpec = new SecretKeySpec(key, ALGORITHM);
        IvParameterSpec ivParameterSpec = new IvParameterSpec(iv);
        
        Cipher cipher = Cipher.getInstance(TRANSFORMATION);
        cipher.init(Cipher.DECRYPT_MODE, secretKeySpec, ivParameterSpec);
        
        byte[] decryptedBytes = cipher.doFinal(Base64.decode(encryptedData, Base64.DEFAULT));
        
        return new String(decryptedBytes);
    }
}