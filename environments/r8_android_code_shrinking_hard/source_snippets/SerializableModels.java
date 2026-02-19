// These model classes are serialized to disk for caching and sent over the network
// They must maintain serialization compatibility across app versions
package com.example.myapp.model;

import java.io.Serializable;

public class UserProfile implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private String userId;
    private String name;
    private String email;
    private long timestamp;
    
    public UserProfile() {
        // Default constructor required for serialization
    }
    
    public UserProfile(String userId, String name, String email, long timestamp) {
        this.userId = userId;
        this.name = name;
        this.email = email;
        this.timestamp = timestamp;
    }
    
    public String getUserId() {
        return userId;
    }
    
    public void setUserId(String userId) {
        this.userId = userId;
    }
    
    public String getName() {
        return name;
    }
    
    public void setName(String name) {
        this.name = name;
    }
    
    public String getEmail() {
        return email;
    }
    
    public void setEmail(String email) {
        this.email = email;
    }
    
    public long getTimestamp() {
        return timestamp;
    }
    
    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }
}

class CacheEntry implements Serializable {
    private static final long serialVersionUID = 2L;
    
    private String key;
    private byte[] data;
    private long expiryTime;
    
    public CacheEntry() {
        // Default constructor required for serialization
    }
    
    public CacheEntry(String key, byte[] data, long expiryTime) {
        this.key = key;
        this.data = data;
        this.expiryTime = expiryTime;
    }
    
    public String getKey() {
        return key;
    }
    
    public void setKey(String key) {
        this.key = key;
    }
    
    public byte[] getData() {
        return data;
    }
    
    public void setData(byte[] data) {
        this.data = data;
    }
    
    public long getExpiryTime() {
        return expiryTime;
    }
    
    public void setExpiryTime(long expiryTime) {
        this.expiryTime = expiryTime;
    }
}

class AppSettings implements Serializable {
    private static final long serialVersionUID = 3L;
    
    private boolean darkMode;
    private int fontSize;
    private String language;
    
    public AppSettings() {
        // Default constructor required for serialization
    }
    
    public AppSettings(boolean darkMode, int fontSize, String language) {
        this.darkMode = darkMode;
        this.fontSize = fontSize;
        this.language = language;
    }
    
    public boolean isDarkMode() {
        return darkMode;
    }
    
    public void setDarkMode(boolean darkMode) {
        this.darkMode = darkMode;
    }
    
    public int getFontSize() {
        return fontSize;
    }
    
    public void setFontSize(int fontSize) {
        this.fontSize = fontSize;
    }
    
    public String getLanguage() {
        return language;
    }
    
    public void setLanguage(String language) {
        this.language = language;
    }
}