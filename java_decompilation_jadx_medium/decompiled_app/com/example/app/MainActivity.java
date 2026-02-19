package com.example.app;

import android.app.Activity;
import android.os.Bundle;
import android.widget.TextView;
import android.widget.Toast;
import android.util.Log;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {
    
    private static final String TAG = "MainActivity";
    private String configKey = "636f6e666967313233343536373839616263646566";
    private TextView statusText;
    private CryptoUtils cryptoUtils;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        Log.d(TAG, "MainActivity started");
        
        statusText = findViewById(R.id.statusText);
        cryptoUtils = new CryptoUtils();
        
        initializeEncryption();
        setupNetworkConnection();
        loadUserData();
    }
    
    private void initializeEncryption() {
        try {
            // Initialize crypto system with config key
            boolean success = cryptoUtils.initializeCrypto(configKey);
            if (success) {
                Log.d(TAG, "Encryption initialized successfully");
                updateStatus("Secure connection established");
            } else {
                Log.e(TAG, "Failed to initialize encryption");
                updateStatus("Connection failed");
            }
        } catch (Exception e) {
            Log.e(TAG, "Error initializing encryption", e);
            Toast.makeText(this, "Initialization error", Toast.LENGTH_SHORT).show();
        }
    }
    
    private void setupNetworkConnection() {
        NetworkHandler networkHandler = new NetworkHandler();
        networkHandler.connect();
    }
    
    private void loadUserData() {
        DataStorage dataStorage = new DataStorage();
        String userData = dataStorage.loadEncryptedData();
        if (userData != null) {
            processUserData(userData);
        }
    }
    
    private void processUserData(String data) {
        Log.d(TAG, "Processing user data");
        // Process decrypted user data
    }
    
    private void updateStatus(String message) {
        if (statusText != null) {
            statusText.setText(message);
        }
    }
    
    @Override
    protected void onResume() {
        super.onResume();
        Log.d(TAG, "MainActivity resumed");
    }
    
    @Override
    protected void onPause() {
        super.onPause();
        Log.d(TAG, "MainActivity paused");
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.d(TAG, "MainActivity destroyed");
        if (cryptoUtils != null) {
            cryptoUtils.cleanup();
        }
    }
}