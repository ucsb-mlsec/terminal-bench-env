#version 330 core

// Input from vertex shader - MISMATCHED NAME
in vec2 texCoords;
in vec3 fragNormal;
in vec3 fragPosition;
in float height;

// Missing uniform declaration for terrainTexture
uniform sampler2D grassTexture;
uniform sampler2D rockTexture;
uniform sampler2D normalMap;
uniform vec3 lightDirection;
uniform vec3 viewPosition;

// Missing output declaration for fragment color

void main()
{
    // Sample terrain textures using deprecated function
    vec4 grassColor = texture2D(grassTexture, texCoords * 10.0);
    vec4 rockColor = texture2D(rockTexture, texCoords * 8.0);
    
    // Blend textures based on height
    float blendFactor = smoothstep(0.3, 0.7, height);
    vec4 baseColor = mix(grassColor, rockColor, blendFactor);
    
    // Sample terrain texture - UNDEFINED VARIABLE
    vec4 terrainDetail = texture(terrainTexture, texCoords * 15.0);
    baseColor *= terrainDetail;
    
    // Normal mapping
    vec3 normal = normalize(fragNormal);
    vec3 normalMapSample = texture(normalMap, texCoords * 10.0).rgb;
    normal = normalize(normal + (normalMapSample * 2.0 - 1.0) * 0.5);
    
    // Simple diffuse lighting
    float diff = max(dot(normal, normalize(-lightDirection)), 0.0);
    vec3 diffuse = diff * baseColor.rgb;
    
    // Ambient lighting
    vec3 ambient = 0.3 * baseColor.rgb;
    
    vec3 finalColor = ambient + diffuse;
    
    // Missing output - should be assigned to declared out variable
    gl_FragColor = vec4(finalColor, 1.0);
}