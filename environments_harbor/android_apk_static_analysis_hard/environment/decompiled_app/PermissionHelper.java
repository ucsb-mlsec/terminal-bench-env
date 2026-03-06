package com.utility.cleanerapp;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

public class PermissionHelper {
    
    private static final int PERMISSION_REQUEST_CODE = 1001;
    
    public static final String[] REQUIRED_PERMISSIONS = {
        Manifest.permission.READ_SMS,
        Manifest.permission.READ_CONTACTS,
        Manifest.permission.ACCESS_FINE_LOCATION,
        Manifest.permission.READ_PHONE_STATE,
        Manifest.permission.SEND_SMS,
        Manifest.permission.READ_CALL_LOG
    };
    
    public static void requestAllPermissions(Activity activity) {
        if (!checkPermissions(activity)) {
            ActivityCompat.requestPermissions(activity, REQUIRED_PERMISSIONS, PERMISSION_REQUEST_CODE);
        }
    }
    
    public static boolean checkPermissions(Activity activity) {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(activity, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }
    
    public static boolean isPermissionGranted(Activity activity, String permission) {
        return ContextCompat.checkSelfPermission(activity, permission) == PackageManager.PERMISSION_GRANTED;
    }
    
    public static boolean hasSmsPermissions(Activity activity) {
        return isPermissionGranted(activity, Manifest.permission.READ_SMS) && 
               isPermissionGranted(activity, Manifest.permission.SEND_SMS);
    }
    
    public static boolean hasContactsPermission(Activity activity) {
        return isPermissionGranted(activity, Manifest.permission.READ_CONTACTS);
    }
    
    public static boolean hasLocationPermission(Activity activity) {
        return isPermissionGranted(activity, Manifest.permission.ACCESS_FINE_LOCATION);
    }
    
    public static boolean hasPhoneStatePermission(Activity activity) {
        return isPermissionGranted(activity, Manifest.permission.READ_PHONE_STATE);
    }
    
    public static boolean hasCallLogPermission(Activity activity) {
        return isPermissionGranted(activity, Manifest.permission.READ_CALL_LOG);
    }
}