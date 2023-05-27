#version 450

layout(location = 0) in vec3 inPos;
layout(location = 2) in vec3 inNormal;

layout(location = 0) out vec3 objectColor;
layout(location = 1) out vec3 lightColor;
layout(location = 2) out vec3 lightPosition;
layout(location = 3) out vec3 normal;
layout(location = 4) out vec3 fragPosition;

layout(push_constant) uniform constants
{
    mat4 modelTransformation;
    vec3 color;
} pushConstants;

layout(binding = 0) uniform UniformBufferObject {
    mat4 proj;
    mat4 view;
} ubo;

void main() {
    gl_Position = ubo.proj * ubo.view * pushConstants.modelTransformation * vec4(inPos, 1.0);
    objectColor = pushConstants.color;
    lightColor = vec3(1.0, 1.0, 1.0);
    lightPosition = vec3(3.0, -3.0, 3.0);
    normal = mat3(transpose(inverse(pushConstants.modelTransformation))) * inNormal;
    fragPosition = vec3(pushConstants.modelTransformation * vec4(inPos, 1.0));
}
