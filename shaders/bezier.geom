#version 150 core

uniform mat4 model;

layout (std140) uniform MatrixData {
	mat4 projector;
	mat4 view;
};


layout(triangles) in;
layout(line_strip, max_vertices = 104) out;

in vec4 ex_Color[];

out vec4 finalColor;

const float interpolationFactor = 0.02;

vec4 Interpolate(vec4 p1, vec4 p2, float percent) {
	vec4 v;
	float diff;

	diff = p2.x - p1.x;
	v.x = p1.x + (diff * percent);
	diff = p2.y - p1.y;
	v.y = p1.y + (diff * percent);
	diff = p2.z - p1.z;
	v.z = p1.z + (diff * percent);

	return v;
}

void EmitCurve(mat4 pvm) {
	float interpolant;
	vec4 i0, i1, o;

	vec4 p0 = gl_in[0].gl_Position;
	vec4 p1 = gl_in[1].gl_Position;
	vec4 p2 = gl_in[2].gl_Position;

	gl_Position = pvm * p2;
	EmitVertex();
	gl_Position = pvm * p1;
	EmitVertex();
	gl_Position = pvm * p0;
	EmitVertex();

	for (interpolant = 0.0; interpolant <= 1.0; interpolant += interpolationFactor) {
		i0 = Interpolate(p0, p1, interpolant);
		i1 = Interpolate(p1, p2, interpolant);

		gl_Position = pvm * Interpolate(i0, i1, interpolant);
		EmitVertex();
	}

	EndPrimitive();
}

void main() {
	mat4 pvm = projector * view * model;

	finalColor = vec4(1.0, 1.0, 1.0, 1.0);

	EmitCurve(pvm);
}