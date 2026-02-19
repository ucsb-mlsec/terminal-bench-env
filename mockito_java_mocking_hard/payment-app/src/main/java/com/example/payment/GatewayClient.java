package com.example.payment;

/**
 * Interface representing an external payment gateway API.
 * This interface will be mocked in tests to simulate various gateway responses
 * without making actual API calls.
 */
public interface GatewayClient {
    
    /**
     * Processes a payment through the external gateway.
     * 
     * @param request The payment request containing transaction details
     * @return PaymentResponse containing the result of the payment attempt
     * @throws GatewayException if the gateway encounters an error during processing
     */
    PaymentResponse processPayment(PaymentRequest request) throws GatewayException;
}