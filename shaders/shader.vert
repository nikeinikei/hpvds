#version 450

layout(location = 0) in vec2 pos;

layout(push_constant) uniform constants
{
    mat4 projMatrix;
	mat4 viewMatrix;
} pushConstants;

layout(location = 0) out vec3 fragColor;

vec3 colors[3] = vec3[](
    vec3(1, 0, 0),
    vec3(0, 1, 0),
    vec3(0, 0, 1)
);

void main() {
    gl_Position = pushConstants.projMatrix * pushConstants.viewMatrix * vec4(pos, 0, 1);
    fragColor = colors[gl_VertexIndex];
}
