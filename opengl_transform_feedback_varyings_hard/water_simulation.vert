#version 330 core

// Input attributes
in vec3 water_pos;
in float wave_amplitude;
in vec2 wave_direction;

// Output variables for transform feedback
out vec3 updated_position;
out vec3 wave_normal;
out float wave_phase;

// Uniform variables
uniform float time;
uniform float frequency;
uniform float dampening;

void main()
{
    // Normalize wave direction
    vec2 dir = normalize(wave_direction);
    
    // Calculate wave phase based on position and time
    float phase_offset = dot(water_pos.xz, dir);
    wave_phase = frequency * time + phase_offset;
    
    // Apply dampening factor
    float damped_amplitude = wave_amplitude * exp(-dampening * time);
    
    // Calculate wave displacement using Gerstner wave equation
    float wave_height = damped_amplitude * sin(wave_phase);
    float wave_height_dx = damped_amplitude * cos(wave_phase);
    
    // Update vertical position
    updated_position = water_pos;
    updated_position.y += wave_height;
    
    // Horizontal displacement for more realistic waves
    updated_position.x += dir.x * wave_height_dx * 0.1;
    updated_position.z += dir.y * wave_height_dx * 0.1;
    
    // Calculate tangent vectors for normal computation
    float k = frequency;
    float dHdx = -k * dir.x * damped_amplitude * cos(wave_phase);
    float dHdz = -k * dir.y * damped_amplitude * cos(wave_phase);
    
    // Compute wave normal using partial derivatives
    vec3 tangent_x = vec3(1.0, dHdx, 0.0);
    vec3 tangent_z = vec3(0.0, dHdz, 1.0);
    wave_normal = normalize(cross(tangent_z, tangent_x));
    
    // Set gl_Position for rasterization pipeline
    gl_Position = vec4(updated_position, 1.0);
}