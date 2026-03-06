package com.example.app;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Base64;
import java.security.MessageDigest;
import javax.crypto.Cipher;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;

public class SecurePreferences {
    private static final String ALGORITHM = "AES/CBC/PKCS5Padding";
    private static final String DIGEST_ALGORITHM = "SHA-256";
    private static String PREF_KEY = "48656c6c6f576f726c64313233343536";
    private static final String ENCODED_SECRET = "cGFzc3dvcmQxMjM0NTY3ODkwYWJjZGVm";
    private static final String PREFS_NAME = "SecurePrefs";
    private Context context;
    private SharedPreferences preferences;
    byte[] prefIV = {(byte)0xFF, (byte)0xEE, (byte)0xDD, (byte)0xCC, (byte)0xBB, (byte)0xAA, (byte)0x99, (byte)0x88, (byte)0x77, (byte)0x66, (byte)0x55, (byte)0x44, (byte)0x33, (byte)0x22, (byte)0x11, (byte)0x00};

    public SecurePreferences(Context context2) {
        this.context = context2;
        this.preferences = context2.getSharedPreferences(PREFS_NAME, 0);
    }

    private byte[] getKeyBytes() {
        try {
            byte[] keyBytes = hexStringToByteArray(PREF_KEY);
            MessageDigest digest = MessageDigest.getInstance(DIGEST_ALGORITHM);
            byte[] hash = digest.digest(keyBytes);
            byte[] key = new byte[16];
            System.arraycopy(hash, 0, key, 0, 16);
            return key;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private byte[] hexStringToByteArray(String str) {
        int len = str.length();
        byte[] data = new byte[(len / 2)];
        for (int i = 0; i < len; i += 2) {
            data[i / 2] = (byte) ((Character.digit(str.charAt(i), 16) << 4) + Character.digit(str.charAt(i + 1), 16));
        }
        return data;
    }

    private String encrypt(String value) {
        try {
            SecretKeySpec keySpec = new SecretKeySpec(getKeyBytes(), "AES");
            IvParameterSpec ivSpec = new IvParameterSpec(this.prefIV);
            Cipher cipher = Cipher.getInstance(ALGORITHM);
            cipher.init(1, keySpec, ivSpec);
            byte[] encrypted = cipher.doFinal(value.getBytes("UTF-8"));
            return Base64.encodeToString(encrypted, 0);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private String decrypt(String encrypted) {
        try {
            SecretKeySpec keySpec = new SecretKeySpec(getKeyBytes(), "AES");
            IvParameterSpec ivSpec = new IvParameterSpec(this.prefIV);
            Cipher cipher = Cipher.getInstance(ALGORITHM);
            cipher.init(2, keySpec, ivSpec);
            byte[] decrypted = cipher.doFinal(Base64.decode(encrypted, 0));
            return new String(decrypted, "UTF-8");
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public String getString(String key, String defaultValue) {
        String encrypted = this.preferences.getString(key, (String) null);
        if (encrypted == null) {
            return defaultValue;
        }
        String decrypted = decrypt(encrypted);
        return decrypted != null ? decrypted : defaultValue;
    }

    public void putString(String key, String value) {
        String encrypted = encrypt(value);
        if (encrypted != null) {
            SharedPreferences.Editor editor = this.preferences.edit();
            editor.putString(key, encrypted);
            editor.apply();
        }
    }

    public int getInt(String key, int defaultValue) {
        String value = getString(key, String.valueOf(defaultValue));
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    public void putInt(String key, int value) {
        putString(key, String.valueOf(value));
    }

    private String getDecodedSecret() {
        return new String(Base64.decode(ENCODED_SECRET, 0));
    }
}