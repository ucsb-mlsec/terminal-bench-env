import numpy as np

class Measurement:
    def __init__(self, sensor_id, readings, timestamp):
        self.sensor_id = sensor_id
        self.readings = readings
        self.timestamp = timestamp