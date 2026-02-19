package com.example.library;

import android.content.Context;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;

/**
 * Utility class for checking network connectivity status.
 * Provides methods to check various types of network connections.
 */
public class NetworkUtils {

    /**
     * Private constructor to prevent instantiation.
     */
    private NetworkUtils() {
        throw new AssertionError("Cannot instantiate utility class");
    }

    /**
     * Checks if any network connection is available.
     *
     * @param context The application context
     * @return true if network is available and connected, false otherwise
     */
    public static boolean isNetworkAvailable(Context context) {
        if (context == null) {
            return false;
        }
        
        ConnectivityManager cm = (ConnectivityManager) context.getSystemService(Context.CONNECTIVITY_SERVICE);
        if (cm != null) {
            NetworkInfo info = cm.getActiveNetworkInfo();
            return info != null && info.isConnected();
        }
        return false;
    }

    /**
     * Checks if WiFi connection is available and connected.
     *
     * @param context The application context
     * @return true if WiFi is connected, false otherwise
     */
    public static boolean isWifiConnected(Context context) {
        if (context == null) {
            return false;
        }
        
        ConnectivityManager cm = (ConnectivityManager) context.getSystemService(Context.CONNECTIVITY_SERVICE);
        if (cm != null) {
            NetworkInfo info = cm.getNetworkInfo(ConnectivityManager.TYPE_WIFI);
            return info != null && info.isConnected();
        }
        return false;
    }

    /**
     * Checks if mobile data connection is available and connected.
     *
     * @param context The application context
     * @return true if mobile data is connected, false otherwise
     */
    public static boolean isMobileConnected(Context context) {
        if (context == null) {
            return false;
        }
        
        ConnectivityManager cm = (ConnectivityManager) context.getSystemService(Context.CONNECTIVITY_SERVICE);
        if (cm != null) {
            NetworkInfo info = cm.getNetworkInfo(ConnectivityManager.TYPE_MOBILE);
            return info != null && info.isConnected();
        }
        return false;
    }
}