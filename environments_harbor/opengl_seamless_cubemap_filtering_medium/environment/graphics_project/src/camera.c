#include <stdio.h>
#include <math.h>
#include <string.h>

// Camera structure definition
// Uses right-handed coordinate system: +X right, +Y up, +Z towards viewer
typedef struct {
    float position[3];  // Camera position in world space
    float target[3];    // Point the camera is looking at
    float up[3];        // Up vector (typically [0, 1, 0])
    float view_matrix[16]; // 4x4 view matrix stored in column-major order
} Camera;

// Global camera instance
static Camera main_camera;

// Helper function: Normalize a 3D vector
static void normalize_vector(float* vec) {
    float length = sqrtf(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
    if (length > 0.0001f) {
        vec[0] /= length;
        vec[1] /= length;
        vec[2] /= length;
    }
}

// Helper function: Calculate cross product of two vectors
// result = a x b
static void cross_product(const float* a, const float* b, float* result) {
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}

// Helper function: Calculate dot product of two vectors
static float dot_product(const float* a, const float* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Initialize camera with default parameters
// Default position: (0, 0, 5) looking at origin (0, 0, 0)
// Up vector: (0, 1, 0) - standard Y-up configuration
void init_camera(void) {
    // Set default position
    main_camera.position[0] = 0.0f;
    main_camera.position[1] = 0.0f;
    main_camera.position[2] = 5.0f;
    
    // Set target to origin
    main_camera.target[0] = 0.0f;
    main_camera.target[1] = 0.0f;
    main_camera.target[2] = 0.0f;
    
    // Set up vector (Y-up)
    main_camera.up[0] = 0.0f;
    main_camera.up[1] = 1.0f;
    main_camera.up[2] = 0.0f;
    
    // Initialize view matrix to identity
    memset(main_camera.view_matrix, 0, sizeof(main_camera.view_matrix));
    main_camera.view_matrix[0] = 1.0f;
    main_camera.view_matrix[5] = 1.0f;
    main_camera.view_matrix[10] = 1.0f;
    main_camera.view_matrix[15] = 1.0f;
    
    update_view_matrix();
}

// Calculate and update the view matrix from camera parameters
// Constructs a look-at matrix using camera position, target, and up vector
void update_view_matrix(void) {
    float forward[3], right[3], up[3];
    
    // Calculate forward vector (from position to target)
    forward[0] = main_camera.target[0] - main_camera.position[0];
    forward[1] = main_camera.target[1] - main_camera.position[1];
    forward[2] = main_camera.target[2] - main_camera.position[2];
    normalize_vector(forward);
    
    // Calculate right vector (forward x up)
    cross_product(forward, main_camera.up, right);
    normalize_vector(right);
    
    // Recalculate up vector (right x forward) to ensure orthogonality
    cross_product(right, forward, up);
    normalize_vector(up);
    
    // Build view matrix (column-major order for OpenGL)
    main_camera.view_matrix[0] = right[0];
    main_camera.view_matrix[1] = up[0];
    main_camera.view_matrix[2] = -forward[0];
    main_camera.view_matrix[3] = 0.0f;
    
    main_camera.view_matrix[4] = right[1];
    main_camera.view_matrix[5] = up[1];
    main_camera.view_matrix[6] = -forward[1];
    main_camera.view_matrix[7] = 0.0f;
    
    main_camera.view_matrix[8] = right[2];
    main_camera.view_matrix[9] = up[2];
    main_camera.view_matrix[10] = -forward[2];
    main_camera.view_matrix[11] = 0.0f;
    
    main_camera.view_matrix[12] = -dot_product(right, main_camera.position);
    main_camera.view_matrix[13] = -dot_product(up, main_camera.position);
    main_camera.view_matrix[14] = dot_product(forward, main_camera.position);
    main_camera.view_matrix[15] = 1.0f;
}

// Rotate camera around the target point
// yaw: rotation around the up axis (horizontal rotation)
// pitch: rotation up/down
void camera_rotate(float yaw, float pitch) {
    float radius, current_yaw, current_pitch;
    float dx, dy, dz;
    
    // Calculate current position relative to target
    dx = main_camera.position[0] - main_camera.target[0];
    dy = main_camera.position[1] - main_camera.target[1];
    dz = main_camera.position[2] - main_camera.target[2];
    
    // Convert to spherical coordinates
    radius = sqrtf(dx * dx + dy * dy + dz * dz);
    current_yaw = atan2f(dx, dz);
    current_pitch = asinf(dy / radius);
    
    // Apply rotation deltas
    current_yaw += yaw;
    current_pitch += pitch;
    
    // Clamp pitch to avoid gimbal lock
    const float max_pitch = 1.5f;
    if (current_pitch > max_pitch) current_pitch = max_pitch;
    if (current_pitch < -max_pitch) current_pitch = -max_pitch;
    
    // Convert back to Cartesian coordinates
    main_camera.position[0] = main_camera.target[0] + radius * sinf(current_pitch) * sinf(current_yaw);
    main_camera.position[1] = main_camera.target[1] + radius * sinf(current_pitch);
    main_camera.position[2] = main_camera.target[2] + radius * cosf(current_pitch) * cosf(current_yaw);
    
    update_view_matrix();
}

// Zoom camera (move closer to or farther from target)
// delta: positive values zoom in, negative values zoom out
void camera_zoom(float delta) {
    float direction[3];
    float distance;
    
    // Calculate direction from position to target
    direction[0] = main_camera.target[0] - main_camera.position[0];
    direction[1] = main_camera.target[1] - main_camera.position[1];
    direction[2] = main_camera.target[2] - main_camera.position[2];
    
    distance = sqrtf(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
    
    // Prevent zooming too close
    if (distance + delta < 0.5f) {
        delta = 0.5f - distance;
    }
    
    normalize_vector(direction);
    
    // Move camera along direction vector
    main_camera.position[0] += direction[0] * delta;
    main_camera.position[1] += direction[1] * delta;
    main_camera.position[2] += direction[2] * delta;
    
    update_view_matrix();
}