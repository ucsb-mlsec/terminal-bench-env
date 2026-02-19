package com.example.network;

import java.net.HttpURLConnection;
import java.net.URL;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.IOException;
import org.json.JSONObject;
import org.json.JSONException;

public class ApiClient {
    private String baseUrl;
    private int timeout;
    private static ApiClient instance;
    
    private ApiClient(String baseUrl) {
        this.baseUrl = baseUrl;
        this.timeout = 30000;
    }
    
    public static ApiClient getInstance(String baseUrl) {
        if (instance == null) {
            synchronized (ApiClient.class) {
                if (instance == null) {
                    instance = new ApiClient(baseUrl);
                }
            }
        }
        return instance;
    }
    
    public String get(String endpoint) throws IOException {
        HttpURLConnection connection = buildConnection(endpoint, "GET");
        return readResponse(connection);
    }
    
    public String post(String endpoint, JSONObject data) throws IOException {
        HttpURLConnection connection = buildConnection(endpoint, "POST");
        connection.setDoOutput(true);
        writeRequestBody(connection, data);
        return readResponse(connection);
    }
    
    public String put(String endpoint, JSONObject data) throws IOException {
        HttpURLConnection connection = buildConnection(endpoint, "PUT");
        connection.setDoOutput(true);
        writeRequestBody(connection, data);
        return readResponse(connection);
    }
    
    public String delete(String endpoint) throws IOException {
        HttpURLConnection connection = buildConnection(endpoint, "DELETE");
        return readResponse(connection);
    }
    
    private HttpURLConnection buildConnection(String endpoint, String method) throws IOException {
        URL url = new URL(baseUrl + endpoint);
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setRequestMethod(method);
        connection.setConnectTimeout(timeout);
        connection.setReadTimeout(timeout);
        connection.setRequestProperty("Content-Type", "application/json");
        connection.setRequestProperty("Accept", "application/json");
        return connection;
    }
    
    private void writeRequestBody(HttpURLConnection connection, JSONObject data) throws IOException {
        OutputStream os = connection.getOutputStream();
        os.write(data.toString().getBytes("UTF-8"));
        os.flush();
        os.close();
    }
    
    private String readResponse(HttpURLConnection connection) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(connection.getInputStream()));
        StringBuilder response = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            response.append(line);
        }
        reader.close();
        return response.toString();
    }
    
    public void setTimeout(int timeout) {
        this.timeout = timeout;
    }
}