#!/usr/bin/env python3

import sys
import time

# Broken producer - does NOT handle SIGPIPE or BrokenPipeError
count = 1
while count <= 10000:
    print(f"Record {count}")
    sys.stdout.flush()
    time.sleep(0.01)
    count += 1