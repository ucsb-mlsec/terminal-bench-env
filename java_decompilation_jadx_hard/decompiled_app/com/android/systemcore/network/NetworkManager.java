package com.android.systemcore.network;

import android.content.Context;
import android.util.Log;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Map;
import org.json.JSONObject;
import com.android.systemcore.config.ApiClient;

public class NetworkManager {
    
    private static final String TAG = "NetworkManager";
    private static NetworkManager instance;
    private Context context;
    private ApiClient apiClient;
    
    private NetworkManager(Context context) {
        this.context = context.getApplicationContext();
        this.apiClient = new ApiClient();
    }
    
    public static synchronized NetworkManager getInstance(Context context) {
        if (instance == null) {
            instance = new NetworkManager(context);
        }
        return instance;
    }
    
    public String sendData(String endpoint, JSONObject data) {
        HttpURLConnection connection = null;
        try {
            String baseUrl = apiClient.getBaseUrl();
            URL url = new URL(baseUrl + endpoint);
            connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("POST");
            connection.setRequestProperty("Content-Type", "application/json");
            connection.setDoOutput(true);
            connection.setConnectTimeout(10000);
            connection.setReadTimeout(10000);
            
            OutputStream os = connection.getOutputStream();
            os.write(data.toString().getBytes("UTF-8"));
            os.flush();
            os.close();
            
            int responseCode = connection.getResponseCode();
            if (responseCode == HttpURLConnection.HTTP_OK) {
                BufferedReader reader = new BufferedReader(
                    new InputStreamReader(connection.getInputStream()));
                StringBuilder response = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) {
                    response.append(line);
                }
                reader.close();
                return response.toString();
            } else {
                Log.e(TAG, "HTTP error code: " + responseCode);
                return null;
            }
        } catch (IOException e) {
            Log.e(TAG, "Network error: " + e.getMessage());
            return null;
        } finally {
            if (connection != null) {
                connection.disconnect();
            }
        }
    }
    
    public boolean uploadInfo(Map<String, String> deviceInfo) {
        try {
            JSONObject jsonData = new JSONObject(deviceInfo);
            String response = sendData("/upload", jsonData);
            return response != null;
        } catch (Exception e) {
            Log.e(TAG, "Upload failed: " + e.getMessage());
            return false;
        }
    }
    
    public void reportStatus(String status) {
        try {
            JSONObject data = new JSONObject();
            data.put("status", status);
            data.put("timestamp", System.currentTimeMillis());
            sendData("/status", data);
        } catch (Exception e) {
            Log.e(TAG, "Status report failed: " + e.getMessage());
        }
    }
}