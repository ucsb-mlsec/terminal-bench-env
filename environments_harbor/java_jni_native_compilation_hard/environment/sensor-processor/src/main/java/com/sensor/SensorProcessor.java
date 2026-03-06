package com.sensor;

public class SensorProcessor {
    static {
        System.loadLibrary("sensor");
    }

    public native int calculateChecksum(int sensorValue);

    public static void main(String[] args) {
        SensorProcessor processor = new SensorProcessor();
        int result = processor.calculateChecksum(42);
        System.out.println("Checksum: " + result);
    }
}