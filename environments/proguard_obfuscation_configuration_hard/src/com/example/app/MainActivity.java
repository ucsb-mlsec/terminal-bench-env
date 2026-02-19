package com.example.app;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;

public class MainActivity extends Activity {
    
    private static final String TAG = "MainActivity";
    private boolean isInitialized = false;
    private Button startButton;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Log.d(TAG, "onCreate called");
        initializeUI();
    }
    
    @Override
    protected void onStart() {
        super.onStart();
        Log.d(TAG, "onStart called");
        isInitialized = true;
    }
    
    @Override
    protected void onResume() {
        super.onResume();
        Log.d(TAG, "onResume called");
        updateUI();
    }
    
    private void initializeUI() {
        startButton = findViewById(R.id.start_button);
        startButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                handleStartClick();
            }
        });
    }
    
    private void updateUI() {
        if (isInitialized) {
            startButton.setEnabled(true);
        }
    }
    
    private void handleStartClick() {
        Log.d(TAG, "Start button clicked");
    }
}