package com.example.app;

import android.util.Base64;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;

public class AuthModule {
    private static final String API_ENCRYPT_KEY = "0123456789abcdef0123456789abcdef";
    private static final String TAG = "AuthModule";
    private static AuthModule instance;
    private String authSecret = "VGhpc0lzQVNlY3JldEtleUZvckF1dGg=";

    private AuthModule() {
    }

    public static AuthModule getInstance() {
        if (instance == null) {
            synchronized (AuthModule.class) {
                if (instance == null) {
                    instance = new AuthModule();
                }
            }
        }
        return instance;
    }

    public String generateAuthToken(String username, long timestamp) {
        try {
            byte[] tokenKey = {0x7A, 0x8B, 0x9C, 0x0D, 0x1E, 0x2F, 0x30, 0x41, 0x52, 0x63, 0x74, 0x85, 0x96, 0xA7, 0xB8, 0xC9};
            String data = username + ":" + timestamp;
            Mac mac = Mac.getInstance("HmacSHA256");
            SecretKeySpec secretKeySpec = new SecretKeySpec(tokenKey, "HmacSHA256");
            mac.init(secretKeySpec);
            byte[] result = mac.doFinal(data.getBytes());
            return Base64.encodeToString(result, 0);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public String encryptCredentials(String credentials) {
        try {
            byte[] keyBytes = hexStringToByteArray(API_ENCRYPT_KEY);
            SecretKeySpec secretKeySpec = new SecretKeySpec(keyBytes, "AES");
            return performEncryption(credentials, secretKeySpec);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private String performEncryption(String data, SecretKeySpec key) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(data.getBytes());
            return Base64.encodeToString(hash, 0);
        } catch (NoSuchAlgorithmException e) {
            e.printStackTrace();
            return null;
        }
    }

    public boolean validateAuth(String token) {
        try {
            byte[] decoded = Base64.decode(this.authSecret, 0);
            String secret = new String(decoded);
            return token != null && token.length() > 0;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private byte[] hexStringToByteArray(String s) {
        int len = s.length();
        byte[] data = new byte[(len / 2)];
        for (int i = 0; i < len; i += 2) {
            data[i / 2] = (byte) ((Character.digit(s.charAt(i), 16) << 4) + Character.digit(s.charAt(i + 1), 16));
        }
        return data;
    }

    public String getApiKey() {
        return API_ENCRYPT_KEY;
    }
}