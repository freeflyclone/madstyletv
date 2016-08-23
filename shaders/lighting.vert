#version 150

layout (std140) uniform ShaderMatrixData {
	mat4 projector;
	mat4 view;
	mat4 model;
};

in vec3 vert;
in vec2 vertTexCoord;
in vec3 vertNormal;
in vec3 vertColor;

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