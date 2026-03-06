package com.example.unused;

// Legacy code - deprecated and scheduled for removal
// This class contains old utility methods that have been replaced
// by more modern implementations in other packages

import java.util.Date;
import java.util.Vector;

/**
 * @deprecated This class is no longer maintained and will be removed in future versions.
 * Use the new utility classes in com.example.utils instead.
 */
public class LegacyCode {
    
    /**
     * Old method for string concatenation
     * @deprecated Use StringBuilder instead
     */
    public static String concatenateStrings(String[] parts) {
        String result = "";
        for (int i = 0; i < parts.length; i++) {
            result = result + parts[i];
        }
        return result;
    }
    
    /**
     * Legacy date formatting method
     * @deprecated Use SimpleDateFormat or DateTimeFormatter
     */
    public static String formatDate(Date date) {
        return date.toString();
    }
    
    /**
     * Old collection utility
     * @deprecated Use ArrayList instead of Vector
     */
    public static Vector createVector(Object[] items) {
        Vector v = new Vector();
        for (int i = 0; i < items.length; i++) {
            v.addElement(items[i]);
        }
        return v;
    }
}