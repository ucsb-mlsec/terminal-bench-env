package com.example.app;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Base64;
import javax.crypto.Cipher;
import javax.crypto.SecretKey;
import javax.crypto.SecretKeyFactory;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.PBEKeySpec;
import javax.crypto.spec.SecretKeySpec;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.security.spec.KeySpec;

public class DataStorage {
    
    private static final String PREFS_NAME = "SecureStorage";
    private static final String STORAGE_KEY = "YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXowMTIzNDU=";
    private Context context;
    
    private byte[] salt = {(byte)0xA9, (byte)0x9B, (byte)0xC8, (byte)0x32, (byte)0x56, (byte)0x35, (byte)0xE3, (byte)0x03};
    
    private String encryptionKey = "deadbeef1234567890abcdef12345678";
    
    private byte[] storageIV = new byte[]{0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F};
    
    public DataStorage(Context context) {
        this.context = context;
    }
    
    public void storeEncryptedData(String key, String data) {
        try {
            SharedPreferences prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
            String encryptedData = encryptData(data);
            prefs.edit().putString(key, encryptedData).apply();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    public String retrieveEncryptedData(String key) {
        try {
            SharedPreferences prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
            String encryptedData = prefs.getString(key, null);
            if (encryptedData != null) {
                return decryptData(encryptedData);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }
    
    private String encryptData(String data) throws Exception {
        byte[] decodedKey = Base64.decode(STORAGE_KEY, Base64.DEFAULT);
        SecretKeySpec keySpec = new SecretKeySpec(decodedKey, "AES");
        IvParameterSpec ivSpec = new IvParameterSpec(storageIV);
        
        Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
        cipher.init(Cipher.ENCRYPT_MODE, keySpec, ivSpec);
        
        byte[] encryptedBytes = cipher.doFinal(data.getBytes("UTF-8"));
        return Base64.encodeToString(encryptedBytes, Base64.DEFAULT);
    }
    
    private String decryptData(String encryptedData) throws Exception {
        byte[] decodedKey = Base64.decode(STORAGE_KEY, Base64.DEFAULT);
        SecretKeySpec keySpec = new SecretKeySpec(decodedKey, "AES");
        IvParameterSpec ivSpec = new IvParameterSpec(storageIV);
        
        Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
        cipher.init(Cipher.DECRYPT_MODE, keySpec, ivSpec);
        
        byte[] encryptedBytes = Base64.decode(encryptedData, Base64.DEFAULT);
        byte[] decryptedBytes = cipher.doFinal(encryptedBytes);
        return new String(decryptedBytes, "UTF-8");
    }
    
    public void storeToFile(String filename, String data) {
        try {
            String encryptedData = encryptDataWithHexKey(data);
            File file = new File(context.getFilesDir(), filename);
            FileOutputStream fos = new FileOutputStream(file);
            fos.write(encryptedData.getBytes("UTF-8"));
            fos.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    private String encryptDataWithHexKey(String data) throws Exception {
        byte[] keyBytes = hexStringToByteArray(encryptionKey);
        SecretKeySpec keySpec = new SecretKeySpec(keyBytes, "AES");
        IvParameterSpec ivSpec = new IvParameterSpec(storageIV);
        
        Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
        cipher.init(Cipher.ENCRYPT_MODE, keySpec, ivSpec);
        
        byte[] encryptedBytes = cipher.doFinal(data.getBytes("UTF-8"));
        return Base64.encodeToString(encryptedBytes, Base64.DEFAULT);
    }
    
    private SecretKey generateKeyFromPassword(String password) throws Exception {
        SecretKeyFactory factory = SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256");
        KeySpec spec = new PBEKeySpec(password.toCharArray(), salt, 65536, 256);
        SecretKey tmp = factory.generateSecret(spec);
        return new SecretKeySpec(tmp.getEncoded(), "AES");
    }
    
    private byte[] hexStringToByteArray(String s) {
        int len = s.length();
        byte[] data = new byte[len / 2];
        for (int i = 0; i < len; i += 2) {
            data[i / 2] = (byte) ((Character.digit(s.charAt(i), 16) << 4)
                    + Character.digit(s.charAt(i + 1), 16));
        }
        return data;
    }
    
    public void clearAllData() {
        SharedPreferences prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        prefs.edit().clear().apply();
    }
}