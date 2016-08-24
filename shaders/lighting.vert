#version 330

layout (std140) uniform ShaderMatrixData {
	mat4 projector;
	mat4 view;
	mat4 model;
};

layout(location=0) in vec3 vert;
layout(location=1) in vec2 vertTexCoord;
layout(location=2) in vec3 vertNormal;
layout(location=3) in vec3 vertColor;

out vec3 fragVert;
out vec2 fragTexCoord;
out vec3 fragNormal;
out vec3 fragColor;

void main() {
	mat4 camera = projector * view;
    // Pass some variables to the fragment shader
    fragTexCoord = vertTexCoord;
    fragNormal = vertNormal;
    fragVert = vert;
	fragColor = vertColor;
    
    // Apply all matrix transformations to vert
    gl_Position = camera * model * vec4(vert, 1);
}