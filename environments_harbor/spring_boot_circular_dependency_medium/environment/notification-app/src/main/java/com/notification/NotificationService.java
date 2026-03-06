package com.notification;

import org.springframework.stereotype.Service;

@Service
public class NotificationService {

    private final UserPreferenceService userPreferenceService;

    public NotificationService(UserPreferenceService userPreferenceService) {
        this.userPreferenceService = userPreferenceService;
    }

    public void sendNotification(String userId, String message) {
        String preferredChannel = userPreferenceService.getPreferredChannel(userId);
        System.out.println("Sending notification to user " + userId + " via " + preferredChannel + ": " + message);
        // Actual notification sending logic would go here
    }

    public void sendBulkNotification(String message) {
        System.out.println("Sending bulk notification: " + message);
        // Bulk notification logic
    }

    public boolean isNotificationEnabled(String userId) {
        return userPreferenceService.isNotificationEnabled(userId);
    }

    public void processNotification(String userId, String message) {
        if (isNotificationEnabled(userId)) {
            sendNotification(userId, message);
        } else {
            System.out.println("Notifications disabled for user: " + userId);
        }
    }
}