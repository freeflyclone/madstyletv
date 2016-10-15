#version 330

layout (std140) uniform MatrixData {
	mat4 projector;
	mat4 view;
	mat4 model;
};

layout (std140) uniform MaterialData {
	vec4 diffuse;
};

layout (std140) uniform LightData {
   vec4 position;
   vec4 color;
} light;

layout(location=0) in vec3 vert;
layout(location=1) in vec2 vertTexCoord;
layout(location=2) in vec3 vertNormal;
layout(location=3) in vec4 vertColor;

out Attributes {
	vec4 color;
} AttributesOut;

void main() {
	// transform normal to camera space and normalize it.
	// Example shader shows tat the transpose()... is passed in as a uniform
	// I'm doing it here because reasons.
    vec3 normal = normalize(transpose(inverse(mat3(model))) * vertNormal);

	float intensity = max(dot(normal, light.position.xyz),0.0);
	AttributesOut.color = intensity * vertColor;

    gl_Position = projector * view * model * vec4(vert, 1);
}
