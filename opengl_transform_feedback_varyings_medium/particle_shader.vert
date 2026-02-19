#version 330 core

// Input attributes from VBO
in vec3 inPosition;
in vec3 inVelocity;
in float inAge;
in vec4 inColor;

// Output variables for transform feedback capture
out vec3 outPosition;        // correct - 3 components
vec3 outVelocity;            // MISSING 'out' qualifier - 3 components
out float outAge;            // correct - 1 component
vec4 outColor;               // MISSING 'out' qualifier - 4 components
out vec2 outSize;            // correct - 2 components
varying float outLifetime;   // WRONG qualifier (varying is GLSL 1.20 syntax) - 1 component
in float outMass;            // WRONG qualifier ('in' instead of 'out') - 1 component

// Simulation parameters (uniforms)
uniform float deltaTime;
uniform vec3 gravity;

void main() {
    // Update position based on velocity
    outPosition = inPosition + inVelocity * deltaTime;
    
    // Update velocity with gravity
    outVelocity = inVelocity + gravity * deltaTime;
    
    // Age the particle
    outAge = inAge + deltaTime;
    
    // Fade color over time
    outColor = inColor * vec4(1.0, 1.0, 1.0, max(0.0, 1.0 - outAge / 5.0));
    
    // Size decreases over time
    outSize = vec2(1.0 - outAge * 0.1);
    
    // Lifetime counter
    outLifetime = 5.0 - outAge;
    
    // Mass stays constant
    outMass = 1.0;
    
    // Set gl_Position for rendering (required for vertex shader)
    gl_Position = vec4(outPosition, 1.0);
}