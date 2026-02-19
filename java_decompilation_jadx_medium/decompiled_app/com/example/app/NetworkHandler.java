package com.example.app;

import android.util.Base64;
import android.util.Log;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.security.MessageDigest;
import javax.crypto.Cipher;
import javax.crypto.SecretKey;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;

public class NetworkHandler {
    private static final String TAG = "NetworkHandler";
    private static final String SECRET_KEY = "aabbccdd1122334455667788aabbccdd";
    private static final String SERVER_URL = "https://api.example.com/data";
    private HttpURLConnection connection;

    public NetworkHandler() {
    }

    private byte[] hexToBytes(String str) {
        int length = str.length();
        byte[] bArr = new byte[(length / 2)];
        for (int i = 0; i < length; i += 2) {
            bArr[i / 2] = (byte) ((Character.digit(str.charAt(i), 16) << 4) + Character.digit(str.charAt(i + 1), 16));
        }
        return bArr;
    }

    public byte[] encryptData(String data) {
        try {
            byte[] keyBytes = hexToBytes(SECRET_KEY);
            SecretKeySpec secretKeySpec = new SecretKeySpec(keyBytes, "AES");
            String hexIV = "0102030405060708090a0b0c0d0e0f10";
            byte[] ivBytes = hexToBytes(hexIV);
            IvParameterSpec ivParameterSpec = new IvParameterSpec(ivBytes);
            Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
            cipher.init(1, secretKeySpec, ivParameterSpec);
            return cipher.doFinal(data.getBytes("UTF-8"));
        } catch (Exception e) {
            Log.e(TAG, "Encryption failed", e);
            return null;
        }
    }

    public String decryptData(byte[] encryptedData) {
        try {
            byte[] keyBytes = hexToBytes(SECRET_KEY);
            SecretKeySpec secretKeySpec = new SecretKeySpec(keyBytes, "AES");
            String hexIV = "0102030405060708090a0b0c0d0e0f10";
            byte[] ivBytes = hexToBytes(hexIV);
            IvParameterSpec ivParameterSpec = new IvParameterSpec(ivBytes);
            Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
            cipher.init(2, secretKeySpec, ivParameterSpec);
            return new String(cipher.doFinal(encryptedData), "UTF-8");
        } catch (Exception e) {
            Log.e(TAG, "Decryption failed", e);
            return null;
        }
    }

    public byte[] encryptWithDES(String data) {
        try {
            byte[] desKey = {0x1A, 0x2B, 0x3C, 0x4D, 0x5E, 0x6F, 0x70, 0x81};
            SecretKeySpec secretKeySpec = new SecretKeySpec(desKey, "DES");
            Cipher cipher = Cipher.getInstance("DES/ECB/PKCS5Padding");
            cipher.init(1, secretKeySpec);
            return cipher.doFinal(data.getBytes("UTF-8"));
        } catch (Exception e) {
            Log.e(TAG, "DES encryption failed", e);
            return null;
        }
    }

    public String sendEncryptedData(String data) {
        try {
            byte[] encryptedData = encryptData(data);
            if (encryptedData == null) {
                return null;
            }
            String encodedData = Base64.encodeToString(encryptedData, 0);
            URL url = new URL(SERVER_URL);
            this.connection = (HttpURLConnection) url.openConnection();
            this.connection.setRequestMethod("POST");
            this.connection.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");
            this.connection.setDoOutput(true);
            DataOutputStream outputStream = new DataOutputStream(this.connection.getOutputStream());
            outputStream.writeBytes("data=" + encodedData);
            outputStream.flush();
            outputStream.close();
            int responseCode = this.connection.getResponseCode();
            if (responseCode == 200) {
                BufferedReader reader = new BufferedReader(new InputStreamReader(this.connection.getInputStream()));
                StringBuilder response = new StringBuilder();
                while (true) {
                    String line = reader.readLine();
                    if (line == null) {
                        break;
                    }
                    response.append(line);
                }
                reader.close();
                return response.toString();
            }
            Log.e(TAG, "Server returned error code: " + responseCode);
            return null;
        } catch (Exception e) {
            Log.e(TAG, "Network request failed", e);
            return null;
        }
    }

    public String receiveEncryptedData() {
        try {
            URL url = new URL(SERVER_URL);
            this.connection = (HttpURLConnection) url.openConnection();
            this.connection.setRequestMethod("GET");
            int responseCode = this.connection.getResponseCode();
            if (responseCode == 200) {
                BufferedReader reader = new BufferedReader(new InputStreamReader(this.connection.getInputStream()));
                StringBuilder response = new StringBuilder();
                while (true) {
                    String line = reader.readLine();
                    if (line == null) {
                        break;
                    }
                    response.append(line);
                }
                reader.close();
                byte[] encryptedData = Base64.decode(response.toString(), 0);
                return decryptData(encryptedData);
            }
            Log.e(TAG, "Server returned error code: " + responseCode);
            return null;
        } catch (Exception e) {
            Log.e(TAG, "Network request failed", e);
            return null;
        }
    }

    public void closeConnection() {
        if (this.connection != null) {
            this.connection.disconnect();
        }
    }
}