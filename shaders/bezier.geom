#version 150 core

uniform mat4 model;

layout (std140) uniform MatrixData {
	mat4 projector;
	mat4 view;
};


layout(points) in;
layout(points, max_vertices = 2) out;

in vec4 ex_Color[];

out vec4 finalColor;

void main() {
	mat4 pvm = projector * view * model;

	gl_Position = pvm * gl_in[0].gl_Position;
	finalColor = vec4(1.0, 1.0, 0.0, 1.0);
	EmitVertex();
	EndPrimitive();

	gl_Position = pvm * (gl_in[0].gl_Position + vec4(0.0, 0.0, 1.0, 0.0));
	finalColor = vec4(0.0, 1.0, 1.0, 1.0);
	EmitVertex();
	EndPrimitive();
}