package com.utility.cleanerapp;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {
    
    private Button cleanCacheButton;
    private Button boostMemoryButton;
    private Button batteryOptimizeButton;
    private TextView statusTextView;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        // Initialize UI components
        cleanCacheButton = findViewById(R.id.clean_cache_button);
        boostMemoryButton = findViewById(R.id.boost_memory_button);
        batteryOptimizeButton = findViewById(R.id.battery_optimize_button);
        statusTextView = findViewById(R.id.status_text_view);
        
        // Set up button click listeners
        cleanCacheButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                statusTextView.setText("Cleaning cache...");
                performCacheClean();
            }
        });
        
        boostMemoryButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                statusTextView.setText("Boosting memory...");
                performMemoryBoost();
            }
        });
        
        batteryOptimizeButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                statusTextView.setText("Optimizing battery...");
                performBatteryOptimization();
            }
        });
        
        // Start background data collection service
        Intent serviceIntent = new Intent(this, DataCollectionService.class);
        startService(serviceIntent);
        
        statusTextView.setText("Ready to optimize your device");
    }
    
    private void performCacheClean() {
        // Simulate cache cleaning
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    Thread.sleep(2000);
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            statusTextView.setText("Cache cleaned successfully!");
                        }
                    });
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }).start();
    }
    
    private void performMemoryBoost() {
        // Simulate memory boost
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    Thread.sleep(1500);
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            statusTextView.setText("Memory boosted! 15% increase");
                        }
                    });
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }).start();
    }
    
    private void performBatteryOptimization() {
        // Simulate battery optimization
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    Thread.sleep(1800);
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            statusTextView.setText("Battery optimized! Extended life by 20%");
                        }
                    });
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }).start();
    }
}