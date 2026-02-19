package com.example.library;

import android.content.Context;
import android.content.SharedPreferences;
import android.preference.PreferenceManager;

/**
 * Helper class for managing SharedPreferences operations.
 * Provides convenient static methods for storing and retrieving
 * different data types from the default SharedPreferences.
 */
public class PreferenceHelper {

    /**
     * Private constructor to prevent instantiation.
     */
    private PreferenceHelper() {
        // Utility class - prevent instantiation
    }

    /**
     * Saves a string value to SharedPreferences.
     *
     * @param context The application context
     * @param key The preference key
     * @param value The string value to save
     */
    public static void saveString(Context context, String key, String value) {
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(context);
        prefs.edit().putString(key, value).apply();
    }

    /**
     * Retrieves a string value from SharedPreferences.
     *
     * @param context The application context
     * @param key The preference key
     * @param defaultValue The default value to return if key doesn't exist
     * @return The stored string value or defaultValue if not found
     */
    public static String getString(Context context, String key, String defaultValue) {
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(context);
        return prefs.getString(key, defaultValue);
    }

    /**
     * Saves a boolean value to SharedPreferences.
     *
     * @param context The application context
     * @param key The preference key
     * @param value The boolean value to save
     */
    public static void saveBoolean(Context context, String key, boolean value) {
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(context);
        prefs.edit().putBoolean(key, value).apply();
    }

    /**
     * Retrieves a boolean value from SharedPreferences.
     *
     * @param context The application context
     * @param key The preference key
     * @param defaultValue The default value to return if key doesn't exist
     * @return The stored boolean value or defaultValue if not found
     */
    public static boolean getBoolean(Context context, String key, boolean defaultValue) {
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(context);
        return prefs.getBoolean(key, defaultValue);
    }

    /**
     * Saves an integer value to SharedPreferences.
     *
     * @param context The application context
     * @param key The preference key
     * @param value The integer value to save
     */
    public static void saveInt(Context context, String key, int value) {
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(context);
        prefs.edit().putInt(key, value).apply();
    }

    /**
     * Retrieves an integer value from SharedPreferences.
     *
     * @param context The application context
     * @param key The preference key
     * @param defaultValue The default value to return if key doesn't exist
     * @return The stored integer value or defaultValue if not found
     */
    public static int getInt(Context context, String key, int defaultValue) {
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(context);
        return prefs.getInt(key, defaultValue);
    }
}