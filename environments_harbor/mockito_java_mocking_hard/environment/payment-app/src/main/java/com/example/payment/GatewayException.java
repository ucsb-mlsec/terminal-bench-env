package com.example.payment;

public class GatewayException extends Exception {
    private String errorCode;

    public GatewayException(String message, String errorCode) {
        super(message);
        this.errorCode = errorCode;
    }

    public GatewayException(String message, String errorCode, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
    }

    public String getErrorCode() {
        return errorCode;
    }
}