package com.example.deprecated;

import java.util.List;
import java.util.ArrayList;

/**
 * This class is deprecated and no longer used in the application.
 * It has been replaced by com.example.utils.ModernHelper.
 * 
 * @deprecated As of version 2.0, replaced by {@link com.example.utils.ModernHelper}
 */
@Deprecated
public class OldHelper {
    
    /**
     * Legacy method for string processing.
     * @deprecated Use ModernHelper.processString() instead
     */
    @Deprecated
    public static String processLegacyString(String input) {
        if (input == null) return "";
        return input.trim().toLowerCase();
    }
    
    /**
     * Legacy method for list conversion.
     * @deprecated Use ModernHelper.convertList() instead
     */
    @Deprecated
    public static List<String> convertToList(String[] array) {
        List<String> result = new ArrayList<>();
        if (array != null) {
            for (String item : array) {
                result.add(item);
            }
        }
        return result;
    }
}