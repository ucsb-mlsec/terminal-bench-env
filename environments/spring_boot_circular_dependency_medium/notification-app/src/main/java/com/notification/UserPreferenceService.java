package com.notification;

import org.springframework.stereotype.Service;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class UserPreferenceService {

    private final NotificationService notificationService;
    private final Map<String, Map<String, String>> preferencesStore;

    public UserPreferenceService(NotificationService notificationService) {
        this.notificationService = notificationService;
        this.preferencesStore = new ConcurrentHashMap<>();
    }

    public Map<String, String> getUserPreferences(String userId) {
        return preferencesStore.getOrDefault(userId, new HashMap<>());
    }

    public void updatePreferences(String userId, Map<String, String> preferences) {
        preferencesStore.put(userId, new HashMap<>(preferences));
        notificationService.sendNotification(userId, "Your preferences have been updated successfully");
    }

    public boolean hasPreference(String userId, String preferenceKey) {
        Map<String, String> userPrefs = getUserPreferences(userId);
        return userPrefs.containsKey(preferenceKey);
    }
}