#!/usr/bin/env python3

import random
from message_callback import message_callback

def generate_rendering_messages():
    """Simulate a graphics rendering pipeline generating debug messages."""
    
    # Message templates for each severity level
    critical_messages = [
        "Shader compilation failed",
        "GPU device lost",
        "Fatal memory allocation error",
        "Critical pipeline state error",
        "Rendering context destroyed",
        "Unrecoverable graphics driver error",
        "Critical texture loading failure",
        "Fatal framebuffer error"
    ]
    
    warning_messages = [
        "Memory allocation warning",
        "Shader performance warning",
        "Texture format conversion",
        "Deprecated API usage detected",
        "Buffer overflow prevented",
        "Suboptimal rendering path",
        "Resource leak detected",
        "Performance degradation warning"
    ]
    
    info_messages = [
        "Frame rendered",
        "Texture loaded successfully",
        "Shader compiled",
        "Buffer allocated",
        "Rendering pass completed",
        "Viewport updated",
        "Camera position changed",
        "Scene graph updated",
        "Material applied",
        "Light source added"
    ]
    
    debug_messages = [
        "Vertex buffer updated",
        "Draw call executed",
        "State change recorded",
        "Uniform variable set",
        "Texture unit bound",
        "Vertex attribute pointer set",
        "Index buffer bound",
        "Framebuffer bound",
        "Blend mode changed",
        "Depth test configured"
    ]
    
    # Generate messages with realistic distribution
    # CRITICAL: ~25 messages
    for i in range(25):
        message = random.choice(critical_messages)
        message_callback("CRITICAL", f"{message} (ID: {i})")
    
    # WARNING: ~25 messages
    for i in range(25):
        message = random.choice(warning_messages)
        message_callback("WARNING", f"{message} (ID: {i})")
    
    # INFO: ~450 messages (accounts for ~47% of total)
    for i in range(450):
        message = random.choice(info_messages)
        message_callback("INFO", f"{message} (Frame: {i})")
    
    # DEBUG: ~500 messages (accounts for ~50% of total)
    for i in range(500):
        message = random.choice(debug_messages)
        message_callback("DEBUG", f"{message} (Call: {i})")

if __name__ == '__main__':
    print("Starting graphics rendering message generation...")
    generate_rendering_messages()
    print("Message generation complete.")