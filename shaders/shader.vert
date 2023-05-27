#version 450

layout(location = 0) in vec3 pos;

layout(push_constant) uniform constants
{
    mat4 modelTransformation;
} pushConstants;

layout(binding = 0) uniform UniformBufferObject {
    mat4 proj;
    mat4 view;
} ubo;

void main() {
    gl_Position = ubo.proj * ubo.view * pushConstants.modelTransformation * vec4(pos, 1.0);
}
