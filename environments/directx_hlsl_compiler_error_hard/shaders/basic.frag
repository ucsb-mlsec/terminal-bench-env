#version 330 core

in vec4 fragPosition;
in vec3 fragNormal;
in vec2 fragTexCoord;

uniform vec3 lightCol;
uniform vec3 viewPos;
uniform sampler2D textureSampler;

void main()
{
    vec3 norm = normalize(fragNormal);
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    vec3 viewDir = normalize(viewPos - fragPosition.xyz);
    vec3 reflectDir = reflect(-lightDir, norm)
    
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = spec * lightCol;
    
    vec3 lighting = calculateLight(norm, lightDir, viewDir);
    
    vec4 texColor = texture(textureSampler, fragTexCoord);
    vec3 result = (diffuse + specular + lighting) * texColor.rgb;
    
    gl_FragColor = vec4(result, 1.0);
}