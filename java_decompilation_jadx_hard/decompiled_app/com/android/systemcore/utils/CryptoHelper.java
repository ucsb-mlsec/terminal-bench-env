package com.android.systemcore.utils;

import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.SecretKeySpec;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.util.Base64;

public class CryptoHelper {
    
    private static final String ALGORITHM = "AES";
    private static final String TRANSFORMATION = "AES/ECB/PKCS5Padding";
    private static final String ENCRYPTION_KEY = "aB9#mK2$pL7@qR4!";
    private static final int KEY_SIZE = 128;
    
    public static String encrypt(String data, String key) {
        try {
            SecretKeySpec secretKey = createSecretKey(key);
            Cipher cipher = Cipher.getInstance(TRANSFORMATION);
            cipher.init(Cipher.ENCRYPT_MODE, secretKey);
            byte[] encryptedBytes = cipher.doFinal(data.getBytes("UTF-8"));
            return encodeBase64(encryptedBytes);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
    
    public static String decrypt(String encryptedData, String key) {
        try {
            SecretKeySpec secretKey = createSecretKey(key);
            Cipher cipher = Cipher.getInstance(TRANSFORMATION);
            cipher.init(Cipher.DECRYPT_MODE, secretKey);
            byte[] decodedBytes = decodeBase64(encryptedData);
            byte[] decryptedBytes = cipher.doFinal(decodedBytes);
            return new String(decryptedBytes, "UTF-8");
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
    
    public static String generateKey() {
        try {
            KeyGenerator keyGenerator = KeyGenerator.getInstance(ALGORITHM);
            keyGenerator.init(KEY_SIZE, new SecureRandom());
            SecretKey secretKey = keyGenerator.generateKey();
            byte[] keyBytes = secretKey.getEncoded();
            return encodeBase64(keyBytes);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
    
    public static String encodeBase64(byte[] data) {
        try {
            return Base64.getEncoder().encodeToString(data);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
    
    public static byte[] decodeBase64(String data) {
        try {
            return Base64.getDecoder().decode(data);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
    
    private static SecretKeySpec createSecretKey(String key) {
        try {
            MessageDigest sha = MessageDigest.getInstance("SHA-256");
            byte[] keyBytes = key.getBytes("UTF-8");
            keyBytes = sha.digest(keyBytes);
            byte[] truncatedKey = new byte[16];
            System.arraycopy(keyBytes, 0, truncatedKey, 0, 16);
            return new SecretKeySpec(truncatedKey, ALGORITHM);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
    
    public static String getDefaultEncryptionKey() {
        return ENCRYPTION_KEY;
    }
    
    public static String encryptWithDefaultKey(String data) {
        return encrypt(data, ENCRYPTION_KEY);
    }
    
    public static String decryptWithDefaultKey(String encryptedData) {
        return decrypt(encryptedData, ENCRYPTION_KEY);
    }
    
    public static byte[] hashData(String data) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            return digest.digest(data.getBytes("UTF-8"));
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}