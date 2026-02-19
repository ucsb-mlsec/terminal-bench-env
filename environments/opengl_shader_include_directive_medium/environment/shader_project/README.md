# Shader Project - Include System Documentation

## Overview

This shader project leverages the OpenGL ARB_shading_language_include extension to enable code sharing across multiple shader stages. The extension allows modular shader development by providing an include mechanism similar to C/C++ preprocessor directives, enabling common utility functions, constants, and mathematical operations to be defined once and reused across vertex, fragment, and geometry shaders.

## Directory Structure

The project follows a hierarchical organization to separate shader implementations from shared code:

```
shader_project/
├── shaders/
│   ├── vertex_shader.glsl
│   ├── fragment_shader.glsl
│   └── geometry_shader.glsl
└── common/
    ├── transforms.glsl
    ├── constants.glsl
    ├── math_helpers.glsl
    └── color_utils.glsl
```

The `shaders/` directory contains the main shader program files for each pipeline stage, while the `common/` directory houses shared utility files that are included by multiple shaders.

## Include Path Convention

The ARB_shading_language_include extension requires adherence to specific path and syntax conventions:

- All include paths must start with `/common/` to reference shared utility files
- Include directives must use double quotes: `#include "/common/filename.glsl"`
- The leading forward slash indicates a path relative to the include root
- Include files should not contain `#version` directives, as these must only appear in the main shader file
- Include directives should appear after the version declaration but before the main shader code

## Expected Files

### Required Shared Files

The following utility files must be present in the `common/` directory:

- **transforms.glsl** - Matrix transformation utilities and coordinate space conversions
- **constants.glsl** - Mathematical and rendering constants (PI, epsilon values, etc.)
- **math_helpers.glsl** - Common mathematical functions and vector operations
- **color_utils.glsl** - Color space conversions and manipulation functions

### Shader Files

The main shader implementations in the `shaders/` directory:

- **vertex_shader.glsl** - Vertex processing stage
- **fragment_shader.glsl** - Fragment/pixel processing stage
- **geometry_shader.glsl** - Optional geometry processing stage

## Troubleshooting

Common issues that prevent the include system from functioning correctly:

- **Missing files**: Ensure all required shared files exist in the `common/` directory
- **Incorrect paths**: Verify include directives reference the correct path (must start with `/common/`)
- **Syntax errors**: Check that include directives use proper syntax with double quotes
- **Version conflicts**: Include files should not contain `#version` directives
- **Case sensitivity**: File paths are case-sensitive on most platforms

When shader compilation fails, carefully review error messages to identify which include file cannot be resolved, then verify the file exists at the expected location with the correct filename.