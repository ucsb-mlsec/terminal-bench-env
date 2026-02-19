package com.example.app.network;

import java.net.HttpURLConnection;
import java.net.URL;
import java.io.OutputStream;
import java.io.InputStream;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import javax.crypto.Cipher;
import javax.crypto.spec.SecretKeySpec;
import javax.crypto.spec.IvParameterSpec;
import android.util.Base64;
import android.os.AsyncTask;
import org.json.JSONObject;
import org.json.JSONException;

public class ApiClient {
    private static final String API_KEY = "abcdef1234567890abcdef1234567890";
    private static final String BASE_URL = "https://api.example.com/v1/";
    private static ApiClient instance;
    
    byte[] signKey = {0x12, 0x34, 0x56, 0x78, (byte)0x90, (byte)0xAB, (byte)0xCD, (byte)0xEF, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, (byte)0x88};
    String headerKey = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo=";
    
    private ApiClient() {
    }
    
    public static synchronized ApiClient getInstance() {
        if (instance == null) {
            instance = new ApiClient();
        }
        return instance;
    }
    
    public void makeRequest(String endpoint, JSONObject data, ApiCallback callback) {
        new RequestTask(endpoint, data, callback).execute();
    }
    
    private String encryptPayload(String payload) {
        try {
            SecretKeySpec keySpec = new SecretKeySpec(API_KEY.getBytes(), "AES");
            Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
            cipher.init(Cipher.ENCRYPT_MODE, keySpec);
            byte[] encrypted = cipher.doFinal(payload.getBytes());
            return Base64.encodeToString(encrypted, Base64.DEFAULT);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
    
    private String decryptResponse(String encrypted) {
        try {
            byte[] decodedKey = Base64.decode(headerKey, Base64.DEFAULT);
            SecretKeySpec keySpec = new SecretKeySpec(decodedKey, 0, 16, "AES");
            Cipher cipher = Cipher.getInstance("AES/ECB/PKCS5Padding");
            cipher.init(Cipher.DECRYPT_MODE, keySpec);
            byte[] decrypted = cipher.doFinal(Base64.decode(encrypted, Base64.DEFAULT));
            return new String(decrypted);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
    
    private byte[] signRequest(byte[] data) {
        byte[] signature = new byte[data.length];
        for (int i = 0; i < data.length; i++) {
            signature[i] = (byte)(data[i] ^ signKey[i % signKey.length]);
        }
        return signature;
    }
    
    private class RequestTask extends AsyncTask<Void, Void, String> {
        private String endpoint;
        private JSONObject data;
        private ApiCallback callback;
        
        RequestTask(String endpoint, JSONObject data, ApiCallback callback) {
            this.endpoint = endpoint;
            this.data = data;
            this.callback = callback;
        }
        
        protected String doInBackground(Void... params) {
            try {
                URL url = new URL(BASE_URL + endpoint);
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("POST");
                conn.setRequestProperty("Content-Type", "application/json");
                conn.setRequestProperty("X-API-Key", API_KEY);
                conn.setDoOutput(true);
                
                String encryptedData = encryptPayload(data.toString());
                OutputStream os = conn.getOutputStream();
                os.write(encryptedData.getBytes());
                os.flush();
                os.close();
                
                int responseCode = conn.getResponseCode();
                if (responseCode == HttpURLConnection.HTTP_OK) {
                    BufferedReader in = new BufferedReader(new InputStreamReader(conn.getInputStream()));
                    String inputLine;
                    StringBuilder response = new StringBuilder();
                    
                    while ((inputLine = in.readLine()) != null) {
                        response.append(inputLine);
                    }
                    in.close();
                    
                    return decryptResponse(response.toString());
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
            return null;
        }
        
        protected void onPostExecute(String result) {
            if (callback != null) {
                if (result != null) {
                    callback.onSuccess(result);
                } else {
                    callback.onError("Request failed");
                }
            }
        }
    }
    
    public interface ApiCallback {
        void onSuccess(String response);
        void onError(String error);
    }
}