package com.example.library;

import android.content.Context;
import android.content.pm.PackageManager;
import android.content.pm.PackageInfo;

/**
 * ContextHelper provides utility methods for working with Android Context.
 * This class offers convenient static methods for common Context-related operations.
 */
public class ContextHelper {
    
    /**
     * Private constructor to prevent instantiation.
     */
    private ContextHelper() {
        // Utility class - do not instantiate
    }
    
    /**
     * Retrieves the version name of the application.
     * 
     * @param context The application context
     * @return The version name of the app, or "unknown" if an error occurs
     */
    public static String getAppVersion(Context context) {
        try {
            PackageManager packageManager = context.getPackageManager();
            PackageInfo packageInfo = packageManager.getPackageInfo(context.getPackageName(), 0);
            return packageInfo.versionName;
        } catch (PackageManager.NameNotFoundException e) {
            return "unknown";
        } catch (Exception e) {
            return "unknown";
        }
    }
    
    /**
     * Retrieves the package name of the application.
     * 
     * @param context The application context
     * @return The package name of the application
     */
    public static String getPackageName(Context context) {
        try {
            return context.getPackageName();
        } catch (Exception e) {
            return "";
        }
    }
    
    /**
     * Checks whether an application with the specified package name is installed.
     * 
     * @param context The application context
     * @param packageName The package name to check
     * @return true if the app is installed, false otherwise
     */
    public static boolean isAppInstalled(Context context, String packageName) {
        try {
            PackageManager packageManager = context.getPackageManager();
            packageManager.getPackageInfo(packageName, 0);
            return true;
        } catch (PackageManager.NameNotFoundException e) {
            return false;
        } catch (Exception e) {
            return false;
        }
    }
}