I need to generate a binary Protocol Buffer file for sensor_001.pb. Let me create the binary content according to the protobuf wire format.

The protobuf wire format for this message with field tags:
- Field 1 (sensor_id, string): tag = 0x0a (field 1, wire type 2)
- Field 2 (timestamp, int64): tag = 0x10 (field 2, wire type 0)
- Field 3 (temperature, float): tag = 0x1d (field 3, wire type 5)
- Field 4 (humidity, float): tag = 0x25 (field 4, wire type 5)

Binary encoding:
- sensor_id "SENSOR_001" (11 bytes): 0x0a 0x0b 53 45 4e 53 4f 52 5f 30 30 31
- timestamp 1704067200: 0x10 0x80 0xc8 0xa8 0xf3 0x06
- temperature 22.5: 0x1d 0x00 0x00 0xb4 0x41
- humidity 45.2: 0x25 0x66 0x66 0x34 0x42


SENSOR_001��¨ó4Bfg4B