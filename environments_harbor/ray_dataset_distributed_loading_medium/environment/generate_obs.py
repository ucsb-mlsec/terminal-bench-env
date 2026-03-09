#!/usr/bin/env python3
"""Generate weather observation JSON files obs_0006.json through obs_1000.json."""
import json
import random
import os

random.seed(42)

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "weather_observations")

stations = ["Station_A", "Station_B", "Station_C", "Station_D", "Station_E", "Station_F"]

for file_num in range(6, 1001):
    num_obs = random.randint(20, 25)
    month = random.randint(1, 12)
    day = random.randint(1, 28)

    observations = []
    for i in range(num_obs):
        hour = i % 24
        minute = 0 if num_obs <= 24 else (i * 30) % 60
        station = stations[i % len(stations)]
        temp = round(random.uniform(18.0, 45.0), 1)
        wind = round(random.uniform(3.0, 40.0), 1)
        humidity = random.randint(25, 90)
        timestamp = f"2024-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00Z"

        observations.append({
            "timestamp": timestamp,
            "temperature": temp,
            "wind_speed": wind,
            "humidity": humidity,
            "location": station
        })

    filename = os.path.join(output_dir, f"obs_{file_num:04d}.json")
    with open(filename, "w") as f:
        json.dump(observations, f, indent=2)

print(f"Generated 995 files (obs_0006.json through obs_1000.json)")
