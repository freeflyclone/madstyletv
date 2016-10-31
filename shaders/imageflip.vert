#version 330

uniform	mat4 model;
uniform int mode;

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
	gl_Position = vec4(in_Position.x, -in_Position.y, in_Position.z, 1.0);

    ex_Color = vec4(in_Color, 1.0);
    UV = in_TexCoord;
}