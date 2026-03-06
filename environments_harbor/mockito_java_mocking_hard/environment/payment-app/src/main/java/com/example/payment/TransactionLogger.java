package com.example.payment;

public interface TransactionLogger {
    void logAttempt(PaymentRequest request, int attemptNumber);
    void logResult(PaymentRequest request, PaymentResponse response, boolean success);
}