package com.android.systemcore;

import android.app.Activity;
import android.os.Bundle;
import android.content.Context;
import android.content.Intent;
import android.util.Log;
import android.view.View;
import android.widget.Toast;

public class MainActivity extends Activity {
    
    private static final String TAG = "MainActivity";
    private Context context;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        this.context = this;
        
        Log.d(TAG, "Initializing application components");
        
        initializeServices();
        
        startBackgroundOperations();
    }
    
    private void initializeServices() {
        try {
            NetworkManager.getInstance(this).initialize();
            Log.d(TAG, "Network manager initialized successfully");
        } catch (Exception e) {
            Log.e(TAG, "Error initializing services", e);
        }
    }
    
    private void startBackgroundOperations() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                NetworkManager.getInstance(context).startCommunication();
            }
        }).start();
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.d(TAG, "Activity destroyed");
    }
}