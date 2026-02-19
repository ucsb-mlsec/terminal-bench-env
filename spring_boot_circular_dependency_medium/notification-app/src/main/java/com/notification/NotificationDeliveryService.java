package com.notification;

import org.springframework.stereotype.Service;

@Service
public class NotificationDeliveryService {

    private final NotificationService notificationService;

    public NotificationDeliveryService(NotificationService notificationService) {
        this.notificationService = notificationService;
    }

    public void deliverEmail(String email, String content) {
        System.out.println("Preparing to deliver email to: " + email);
        notificationService.validateNotification(content);
        System.out.println("Email delivered successfully to: " + email);
        notificationService.logDelivery("EMAIL", email);
    }

    public void deliverSms(String phone, String content) {
        System.out.println("Preparing to deliver SMS to: " + phone);
        notificationService.validateNotification(content);
        System.out.println("SMS delivered successfully to: " + phone);
        notificationService.logDelivery("SMS", phone);
    }

    public void deliverPushNotification(String deviceToken, String content) {
        System.out.println("Preparing to deliver push notification to device: " + deviceToken);
        notificationService.validateNotification(content);
        System.out.println("Push notification delivered successfully");
        notificationService.logDelivery("PUSH", deviceToken);
    }
}