#version 330 core

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

#include "/common/transforms.glsl"
#include "/common/geometry_utils.glsl"

in vec3 fragNormal[];
in vec3 fragPos[];

out vec3 geoNormal;
out vec3 geoPos;

void main()
{
    for(int i = 0; i < 3; i++)
    {
        gl_Position = gl_in[i].gl_Position;
        geoNormal = transformNormal(fragNormal[i]);
        geoPos = fragPos[i];
        
        // Apply geometry transformation
        vec3 offset = calculateGeometryOffset(geoPos, geoNormal);
        gl_Position.xyz += offset;
        
        EmitVertex();
    }
    EndPrimitive();
}