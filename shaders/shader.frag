#version 450

layout(location = 0) in vec3 objectColor;
layout(location = 1) in vec3 lightColor;
layout(location = 2) in vec3 lightPosition;
layout(location = 3) in vec3 normal;
layout(location = 4) in vec3 fragPosition;

layout(location = 0) out vec4 outColor;

void main() {
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(lightPosition - fragPosition);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    vec3 result = (ambient + diffuse) * objectColor;

    outColor = vec4(result, 1.0);
}
