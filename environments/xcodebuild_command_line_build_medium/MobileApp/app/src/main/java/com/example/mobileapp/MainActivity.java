package com.example.mobileapp;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;

/**
 * MainActivity - The main entry point for the Mobile App
 * This is a basic main activity that serves as the primary screen of the application
 */
public class MainActivity extends AppCompatActivity {

    /**
     * Called when the activity is first created
     * Initializes the activity and sets up the main content view
     * 
     * @param savedInstanceState Bundle containing the activity's previously saved state
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }
}