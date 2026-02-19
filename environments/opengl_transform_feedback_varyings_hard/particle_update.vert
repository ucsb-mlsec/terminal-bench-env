#version 330 core

// Input attributes
in vec3 position;
in vec3 velocity;
in float lifetime;

// Output variables for transform feedback
out vec3 out_position;
out vec3 out_velocity;
out float out_lifetime;
out vec4 out_color;

// Uniform variables
uniform float deltaTime;
uniform vec3 gravity;

void main()
{
    // Update position based on velocity and time
    out_position = position + velocity * deltaTime;
    
    // Apply gravity to velocity
    out_velocity = velocity + gravity * deltaTime;
    
    // Decrement lifetime
    out_lifetime = lifetime - deltaTime;
    
    // Calculate color based on lifetime (fading effect)
    // Assume particles start with lifetime of 1.0 and fade to 0.0
    float normalizedLifetime = clamp(out_lifetime, 0.0, 1.0);
    
    // Red to yellow to transparent fade
    out_color = vec4(1.0, normalizedLifetime, 0.0, normalizedLifetime);
    
    // Set vertex position for rasterization (if needed)
    gl_Position = vec4(out_position, 1.0);
}