package com.example.payment;

public class PaymentRequest {
    private String transactionId;
    private double amount;
    private String currency;
    private String customerId;

    public PaymentRequest(String transactionId, double amount, String currency, String customerId) {
        this.transactionId = transactionId;
        this.amount = amount;
        this.currency = currency;
        this.customerId = customerId;
    }

    public String getTransactionId() {
        return transactionId;
    }

    public double getAmount() {
        return amount;
    }

    public String getCurrency() {
        return currency;
    }

    public String getCustomerId() {
        return customerId;
    }

    @Override
    public String toString() {
        return "PaymentRequest{" +
                "transactionId='" + transactionId + '\'' +
                ", amount=" + amount +
                ", currency='" + currency + '\'' +
                ", customerId='" + customerId + '\'' +
                '}';
    }
}