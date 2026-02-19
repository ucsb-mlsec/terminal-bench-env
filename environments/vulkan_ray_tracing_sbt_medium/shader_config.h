#ifndef SHADER_CONFIG_H
#define SHADER_CONFIG_H

#include <vulkan/vulkan.h>
#include <cstdint>

// Shader Configuration for Vulkan Ray Tracing Application
// This header defines the shader group layout for the Shader Binding Table (SBT)

// ============================================================================
// Shader Group Counts
// ============================================================================
// Each shader type may have multiple groups in the SBT

#define RAYGEN_SHADER_COUNT 1      // Single ray generation shader
#define MISS_SHADER_COUNT 2        // Primary miss + shadow miss shaders
#define HIT_SHADER_COUNT 3         // Multiple hit groups for different geometry
#define CALLABLE_SHADER_COUNT 2    // Utility callable shaders

#define TOTAL_SHADER_GROUPS 8      // Sum of all shader groups

// ============================================================================
// Shader Record Data Sizes (bytes)
// ============================================================================
// These define additional data stored alongside each shader handle in the SBT
// This data is accessible in shaders and typically contains material parameters,
// texture indices, or other per-shader-group configuration

#define RAYGEN_RECORD_SIZE 0       // Ray gen has no additional data
#define MISS_RECORD_SIZE 8         // Miss shaders store background color data
#define HIT_RECORD_SIZE 16         // Hit shaders store material parameters
#define CALLABLE_RECORD_SIZE 0     // Callable shaders have no extra data

// ============================================================================
// Shader Group Type Enumeration
// ============================================================================
// Identifies the type of each shader group in the pipeline

enum ShaderGroupType {
    RAY_GEN_GROUP = 0,    // Ray generation shader group
    MISS_GROUP = 1,       // Miss shader group
    HIT_GROUP = 2,        // Hit group (closest hit + any hit)
    CALLABLE_GROUP = 3    // Callable shader group
};

// ============================================================================
// Shader Group Index Layout
// ============================================================================
// Defines the ordering of shader groups within the SBT
// 
// SBT Layout:
// [0]     - Ray Generation shader
// [1-2]   - Miss shaders (primary, shadow)
// [3-5]   - Hit groups (geometry type 0, 1, 2)
// [6-7]   - Callable shaders (utility functions)

#define RAYGEN_INDEX_START 0
#define MISS_INDEX_START 1
#define HIT_INDEX_START 3
#define CALLABLE_INDEX_START 6

// ============================================================================
// Shader Group Descriptions
// ============================================================================
// Ray Gen: Generates primary rays for the scene
// Miss Shaders: Handle rays that don't hit geometry (sky, shadows)
// Hit Groups: Handle ray-geometry intersections (materials, lighting)
// Callable: Reusable shader functions (noise, utility calculations)

#endif // SHADER_CONFIG_H