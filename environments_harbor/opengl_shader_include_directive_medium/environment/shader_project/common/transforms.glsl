// Transform utility functions for shader include system
// This file provides common transformation operations shared across shaders

// Apply a 4x4 transformation matrix to a 3D position vector
// Returns the transformed position as a vec3
vec3 applyTransform(vec3 position, mat4 transform)
{
    return (transform * vec4(position, 1.0)).xyz;
}

// Transform and normalize a normal vector using a normal matrix
// The normal matrix is typically the inverse transpose of the model matrix
// Returns the transformed and normalized normal vector
vec3 computeNormal(vec3 normal, mat4 normalMatrix)
{
    return normalize((normalMatrix * vec4(normal, 0.0)).xyz);
}

// Build a rotation matrix around an arbitrary axis
// angle: rotation angle in radians
// axis: normalized rotation axis vector
// Returns a 4x4 rotation matrix
mat4 buildRotationMatrix(float angle, vec3 axis)
{
    float c = cos(angle);
    float s = sin(angle);
    float t = 1.0 - c;
    vec3 a = normalize(axis);
    
    return mat4(
        t * a.x * a.x + c,      t * a.x * a.y + s * a.z, t * a.x * a.z - s * a.y, 0.0,
        t * a.x * a.y - s * a.z, t * a.y * a.y + c,      t * a.y * a.z + s * a.x, 0.0,
        t * a.x * a.z + s * a.y, t * a.y * a.z - s * a.x, t * a.z * a.z + c,      0.0,
        0.0,                    0.0,                    0.0,                    1.0
    );
}