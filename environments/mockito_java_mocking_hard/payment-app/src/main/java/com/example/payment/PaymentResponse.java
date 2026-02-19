package com.example.payment;

public class PaymentResponse {
    private final String responseId;
    private final boolean success;
    private final String errorCode;
    private final String message;

    public PaymentResponse(String responseId, boolean success, String errorCode, String message) {
        this.responseId = responseId;
        this.success = success;
        this.errorCode = errorCode;
        this.message = message;
    }

    public String getResponseId() {
        return responseId;
    }

    public boolean isSuccess() {
        return success;
    }

    public String getErrorCode() {
        return errorCode;
    }

    public String getMessage() {
        return message;
    }

    @Override
    public String toString() {
        return "PaymentResponse{" +
                "responseId='" + responseId + '\'' +
                ", success=" + success +
                ", errorCode='" + errorCode + '\'' +
                ", message='" + message + '\'' +
                '}';
    }
}