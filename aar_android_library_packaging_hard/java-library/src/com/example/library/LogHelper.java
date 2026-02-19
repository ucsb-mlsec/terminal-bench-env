/**
 * LogHelper - Utility class for Android logging operations.
 * 
 * This class provides convenient static methods for logging debug, info,
 * and error messages using Android's Log utility.
 * 
 * @author Example Library Team
 * @version 1.0
 */
package com.example.library;

import android.util.Log;

/**
 * A utility class that provides simplified logging methods for Android applications.
 * This class wraps the standard Android Log class with convenient static methods.
 */
public class LogHelper {
    
    /**
     * Private constructor to prevent instantiation of this utility class.
     * This class is designed to be used through its static methods only.
     */
    private LogHelper() {
        throw new AssertionError("LogHelper is a utility class and should not be instantiated");
    }
    
    /**
     * Logs a debug message to the Android log output.
     * 
     * @param tag Used to identify the source of a log message. It usually identifies
     *            the class or activity where the log call occurs.
     * @param message The message you would like logged.
     */
    public static void debug(String tag, String message) {
        Log.d(tag, message);
    }
    
    /**
     * Logs an informational message to the Android log output.
     * 
     * @param tag Used to identify the source of a log message. It usually identifies
     *            the class or activity where the log call occurs.
     * @param message The message you would like logged.
     */
    public static void info(String tag, String message) {
        Log.i(tag, message);
    }
    
    /**
     * Logs an error message to the Android log output.
     * 
     * @param tag Used to identify the source of a log message. It usually identifies
     *            the class or activity where the log call occurs.
     * @param message The message you would like logged.
     */
    public static void error(String tag, String message) {
        Log.e(tag, message);
    }
}