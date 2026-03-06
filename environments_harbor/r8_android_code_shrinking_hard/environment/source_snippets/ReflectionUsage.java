package com.example.myapp.reflection;

import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;

/**
 * ReflectionUsage - Handles dynamic class loading and method invocation
 * Used for plugin system and runtime configuration
 */
public class ReflectionUsage {
    
    private Map<String, Object> pluginCache = new HashMap<>();
    
    /**
     * Loads plugin classes dynamically based on configuration
     * Plugins are not known at compile time and are loaded from external sources
     */
    public Object loadPlugin(String pluginType) {
        try {
            String className;
            if (pluginType.equals("handler")) {
                className = "com.example.plugins.PluginHandler";
            } else if (pluginType.equals("processor")) {
                className = "com.example.plugins.DataProcessor";
            } else {
                className = "com.example.plugins.DefaultPlugin";
            }
            
            // Load class dynamically - class name comes from configuration
            Class<?> pluginClass = Class.forName(className);
            Object pluginInstance = pluginClass.newInstance();
            pluginCache.put(pluginType, pluginInstance);
            
            return pluginInstance;
        } catch (ClassNotFoundException e) {
            System.err.println("Plugin class not found: " + e.getMessage());
            return null;
        } catch (InstantiationException e) {
            System.err.println("Cannot instantiate plugin: " + e.getMessage());
            return null;
        } catch (IllegalAccessException e) {
            System.err.println("Cannot access plugin constructor: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * Creates objects dynamically using Constructor.newInstance()
     * Used for dependency injection and factory patterns
     */
    public Object createInstance(String className, Object... args) {
        try {
            Class<?> clazz = Class.forName(className);
            
            if (args.length == 0) {
                return clazz.newInstance();
            } else {
                // Find constructor matching argument types
                Class<?>[] paramTypes = new Class<?>[args.length];
                for (int i = 0; i < args.length; i++) {
                    paramTypes[i] = args[i].getClass();
                }
                
                Constructor<?> constructor = clazz.getDeclaredConstructor(paramTypes);
                constructor.setAccessible(true);
                return constructor.newInstance(args);
            }
        } catch (Exception e) {
            System.err.println("Failed to create instance: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * Invokes methods dynamically by name
     * Used for lifecycle callbacks and event handling from configuration
     */
    public Object invokeMethod(Object target, String methodName, Object... args) {
        try {
            Class<?> targetClass = target.getClass();
            
            // Build parameter types array
            Class<?>[] paramTypes = new Class<?>[args.length];
            for (int i = 0; i < args.length; i++) {
                paramTypes[i] = args[i].getClass();
            }
            
            // Find and invoke the method
            Method method = targetClass.getDeclaredMethod(methodName, paramTypes);
            method.setAccessible(true);
            return method.invoke(target, args);
            
        } catch (NoSuchMethodException e) {
            System.err.println("Method not found: " + methodName);
            return null;
        } catch (Exception e) {
            System.err.println("Method invocation failed: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * Calls standard lifecycle methods on plugin instances
     * Method names are standardized: initialize, process, execute
     */
    public void initializePlugin(Object plugin, Map<String, Object> config) {
        try {
            // Call initialize method with configuration
            Method initMethod = plugin.getClass().getDeclaredMethod("initialize", Map.class);
            initMethod.setAccessible(true);
            initMethod.invoke(plugin, config);
            
            // Call process method if available
            try {
                Method processMethod = plugin.getClass().getDeclaredMethod("process");
                processMethod.setAccessible(true);
                processMethod.invoke(plugin);
            } catch (NoSuchMethodException e) {
                // Process method is optional
            }
            
            // Call execute method if available
            try {
                Method executeMethod = plugin.getClass().getDeclaredMethod("execute");
                executeMethod.setAccessible(true);
                executeMethod.invoke(plugin);
            } catch (NoSuchMethodException e) {
                // Execute method is optional
            }
            
        } catch (Exception e) {
            System.err.println("Plugin initialization failed: " + e.getMessage());
        }
    }
    
    /**
     * Accesses private fields for debugging and runtime inspection
     * Used by diagnostic tools and configuration overrides
     */
    public Object getFieldValue(Object target, String fieldName) {
        try {
            Class<?> targetClass = target.getClass();
            Field field = targetClass.getDeclaredField(fieldName);
            field.setAccessible(true);
            return field.get(target);
        } catch (NoSuchFieldException e) {
            System.err.println("Field not found: " + fieldName);
            return null;
        } catch (IllegalAccessException e) {
            System.err.println("Cannot access field: " + fieldName);
            return null;
        }
    }
    
    /**
     * Sets private field values dynamically
     * Used for dependency injection and test configuration
     */
    public void setFieldValue(Object target, String fieldName, Object value) {
        try {
            Class<?> targetClass = target.getClass();
            Field field = targetClass.getDeclaredField(fieldName);
            field.setAccessible(true);
            field.set(target, value);
        } catch (Exception e) {
            System.err.println("Failed to set field value: " + e.getMessage());
        }
    }
}