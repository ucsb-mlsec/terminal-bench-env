package com.example.utils;

import java.util.logging.Level;
import java.util.logging.Logger as JLogger;

public class Logger {
    
    private static final String TAG = "AppLogger";
    private static final JLogger logger = JLogger.getLogger(TAG);
    
    private Logger() {
        // Private constructor to prevent instantiation
    }
    
    public static void debug(String message) {
        logger.log(Level.FINE, formatMessage(message));
    }
    
    public static void info(String message) {
        logger.log(Level.INFO, formatMessage(message));
    }
    
    public static void warn(String message) {
        logger.log(Level.WARNING, formatMessage(message));
    }
    
    public static void error(String message) {
        logger.log(Level.SEVERE, formatMessage(message));
    }
    
    public static void error(String message, Throwable throwable) {
        logger.log(Level.SEVERE, formatMessage(message), throwable);
    }
    
    private static String formatMessage(String message) {
        return "[" + TAG + "] " + message;
    }
    
    public static void setLevel(Level level) {
        logger.setLevel(level);
    }
}