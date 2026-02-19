syntax = "proto3";

message SensorReading {
  string sensor_id = 1;
  int64 timestamp = 2;
  float temperature = 3;
  float humidity = 4;
}