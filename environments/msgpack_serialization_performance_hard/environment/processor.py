#!/usr/bin/env python3

import json
import msgpack
import time

def process_events():
    # Inefficient implementation - deliberately slow
    events = []
    
    # Read file line by line unnecessarily
    with open('events.json', 'r') as f:
        content = f.read()
    
    # Parse JSON
    data = json.loads(content)
    
    # Process each event inefficiently
    for event in data:
        # Create unnecessary temporary copies
        temp_event = {}
        for key in event.keys():
            temp_event[key] = event[key]
        
        # More unnecessary operations
        serialized = json.dumps(temp_event)
        deserialized = json.loads(serialized)
        
        # Append to list
        events.append(deserialized)
    
    # Create more unnecessary copies
    final_events = []
    for e in events:
        event_copy = {}
        for k, v in e.items():
            event_copy[k] = v
        final_events.append(event_copy)
    
    # Serialize to MessagePack inefficiently
    temp_packs = []
    for event in final_events:
        packed = msgpack.packb(event)
        temp_packs.append(packed)
    
    # Unpack and repack all together (very inefficient)
    unpacked_events = []
    for packed in temp_packs:
        unpacked = msgpack.unpackb(packed, raw=False)
        unpacked_events.append(unpacked)
    
    # Finally write to file
    with open('events.msgpack', 'wb') as f:
        msgpack.pack(unpacked_events, f)

if __name__ == '__main__':
    process_events()