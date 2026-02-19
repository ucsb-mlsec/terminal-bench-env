package com.example.payment;

public class PaymentService {
    private final GatewayClient gatewayClient;
    private final TransactionLogger logger;
    private final int maxRetries;

    public PaymentService(GatewayClient gatewayClient, TransactionLogger logger) {
        this.gatewayClient = gatewayClient;
        this.logger = logger;
        this.maxRetries = 3;
    }

    public PaymentResponse processPayment(PaymentRequest request) {
        PaymentResponse response = null;
        
        // BUG 1: Loop starts at 1 instead of 0, causing one fewer retry than maxRetries
        for (int i = 1; i <= maxRetries; i++) {
            logger.logAttempt(request.getTransactionId(), i);
            
            try {
                response = gatewayClient.processPayment(request);
                
                if (response.isSuccess()) {
                    logger.logResult(request.getTransactionId(), "SUCCESS", response.getResponseId());
                    return response;
                }
                
            } catch (GatewayException e) {
                String errorCode = e.getErrorCode();
                
                // Retryable errors
                if ("RATE_LIMIT".equals(errorCode) || "TIMEOUT".equals(errorCode)) {
                    // Continue to next retry
                    continue;
                }
                
                // BUG 2: Checking for 'INSUFFICIENT_FUND' (missing S) instead of 'INSUFFICIENT_FUNDS'
                // This causes INSUFFICIENT_FUNDS errors to retry instead of failing immediately
                if ("INSUFFICIENT_FUND".equals(errorCode) || "INVALID_CARD".equals(errorCode) || "FRAUD_DETECTED".equals(errorCode)) {
                    // Should fail immediately
                    response = new PaymentResponse(
                        "FAILED",
                        false,
                        errorCode,
                        e.getMessage()
                    );
                    logger.logResult(request.getTransactionId(), "FAILED", errorCode);
                    return response;
                }
                
                // Other errors, continue retrying
            }
        }
        
        // BUG 3: logger.logResult() is not called here when all retries are exhausted
        // All retries exhausted
        return new PaymentResponse(
            "FAILED",
            false,
            "MAX_RETRIES_EXCEEDED",
            "Payment failed after retries"
        );
    }
}