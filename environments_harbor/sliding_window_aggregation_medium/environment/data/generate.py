#!/usr/bin/env python3
"""Generate additional stock price data to reach ~10000 entries."""
import random

data_path = "/wekafs/kazhu/code/terminal-bench-env/environments_harbor/sliding_window_aggregation_medium/environment/data/stock_prices.txt"

# Read existing data to find last timestamp and price
data_lines = []
with open(data_path, 'r') as f:
    for line in f:
        line = line.rstrip()
        if line and ',' in line:
            data_lines.append(line)

print(f"Existing data lines: {len(data_lines)}")
last_ts = int(data_lines[-1].split(',')[0])
last_price = float(data_lines[-1].split(',')[1])
print(f"Last timestamp: {last_ts}, Last price: {last_price}")

target = 10000
needed = target - len(data_lines)
print(f"Need to generate {needed} more entries")

random.seed(42)
new_lines = []
current_price = last_price
ts = last_ts + 1

for i in range(needed):
    # Create alternating high/low volatility regions (every 200 entries)
    region = (i // 200) % 6

    if region in (0, 3):
        # Low volatility: small changes
        change = random.uniform(-2.0, 2.0)
    elif region in (1, 4):
        # High volatility: large swings
        change = random.uniform(-15.0, 15.0)
    else:
        # Medium volatility
        change = random.uniform(-5.0, 5.0)

    current_price += change

    # Keep price within 150-450 range
    if current_price < 150.0:
        current_price = 150.0 + random.uniform(10, 30)
    elif current_price > 450.0:
        current_price = 450.0 - random.uniform(10, 30)

    new_lines.append(f"{ts},{round(current_price, 2)}")
    ts += 1

print(f"Generated {len(new_lines)} new lines")

# Append to the data file
with open(data_path, 'a') as f:
    for line in new_lines:
        f.write(line + '\n')

print("Done!")
