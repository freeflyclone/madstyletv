#version 150

uniform mat4 model;

layout (std140) uniform MatrixData {
	mat4 projector;
	mat4 view;
};


in  vec3 in_Position;
in  vec2 in_TexCoord;
in  vec3 in_Normal;
in  vec3 in_Color;

out vec4 ex_Color;
out vec2 UV;

void main(void)
{
    gl_Position = projector * view * model * vec4(in_Position, 1.0);
    ex_Color = vec4(in_Color, 1.0);
    UV = in_TexCoord;
}