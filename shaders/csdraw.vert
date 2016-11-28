#version 430

uniform mat4 model;

layout (std140) uniform MatrixData {
	mat4 projector;
	mat4 view;
};

in  vec3 in_Position;
in  vec2 in_TexCoord;
in  vec3 in_Normal;
in  vec4 in_Color;

out vec4 out_Color;
out vec2 texCoord;

void main() {
	texCoord = vec2(in_Position.x, in_Position.y) * 0.5f + 0.5f;
	out_Color = in_Color;
	gl_Position = projector * view * model * vec4(in_Position, 1.0);
}
