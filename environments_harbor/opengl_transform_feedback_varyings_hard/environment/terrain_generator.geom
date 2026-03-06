#version 330 core

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in vec3 world_position[];

out vec3 terrain_position;
out vec3 terrain_normal;
out vec2 terrain_uv;
out float terrain_height;
out vec4 terrain_color;

uniform float gridSize;
uniform sampler2D heightmap;
uniform mat4 viewProjection;

float sampleHeight(vec2 pos) {
    vec2 uv = pos * 0.01 + 0.5;
    return texture(heightmap, uv).r * 50.0;
}

vec3 calculateNormal(vec3 p0, vec3 p1, vec3 p2) {
    vec3 v1 = p1 - p0;
    vec3 v2 = p2 - p0;
    return normalize(cross(v1, v2));
}

vec4 getTerrainColor(float height) {
    if (height < 5.0) {
        return vec4(0.2, 0.4, 0.8, 1.0);
    } else if (height < 15.0) {
        return vec4(0.8, 0.7, 0.5, 1.0);
    } else if (height < 30.0) {
        return vec4(0.2, 0.6, 0.2, 1.0);
    } else {
        return vec4(0.9, 0.9, 0.9, 1.0);
    }
}

void main() {
    vec3 basePos = world_position[0];
    float halfGrid = gridSize * 0.5;
    
    vec3 corners[4];
    corners[0] = basePos + vec3(-halfGrid, 0.0, -halfGrid);
    corners[1] = basePos + vec3(halfGrid, 0.0, -halfGrid);
    corners[2] = basePos + vec3(-halfGrid, 0.0, halfGrid);
    corners[3] = basePos + vec3(halfGrid, 0.0, halfGrid);
    
    for (int i = 0; i < 4; i++) {
        corners[i].y = sampleHeight(corners[i].xz);
    }
    
    vec3 normal = calculateNormal(corners[0], corners[1], corners[2]);
    
    terrain_position = corners[0];
    terrain_normal = normal;
    terrain_uv = vec2(0.0, 0.0);
    terrain_height = corners[0].y;
    terrain_color = getTerrainColor(terrain_height);
    gl_Position = viewProjection * vec4(corners[0], 1.0);
    EmitVertex();
    
    terrain_position = corners[1];
    terrain_normal = normal;
    terrain_uv = vec2(1.0, 0.0);
    terrain_height = corners[1].y;
    terrain_color = getTerrainColor(terrain_height);
    gl_Position = viewProjection * vec4(corners[1], 1.0);
    EmitVertex();
    
    terrain_position = corners[2];
    terrain_normal = normal;
    terrain_uv = vec2(0.0, 1.0);
    terrain_height = corners[2].y;
    terrain_color = getTerrainColor(terrain_height);
    gl_Position = viewProjection * vec4(corners[2], 1.0);
    EmitVertex();
    
    terrain_position = corners[3];
    terrain_normal = normal;
    terrain_uv = vec2(1.0, 1.0);
    terrain_height = corners[3].y;
    terrain_color = getTerrainColor(terrain_height);
    gl_Position = viewProjection * vec4(corners[3], 1.0);
    EmitVertex();
    
    EndPrimitive();
}