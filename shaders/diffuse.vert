#version 150

layout (std140) uniform ShaderMatrixData {
	mat4 projector;
	mat4 view;
	mat4 model;
};

in vec3 vert;
in vec2 vertTexCoord;
in vec3 vertNormal;
in vec4 vertColor;

out vec3 fragVert;
out vec2 fragTexCoord;
out vec3 fragNormal;
out vec4 fragColor;

void main() {
    // Pass some variables to the fragment shader
    fragTexCoord = vertTexCoord;
    fragNormal = vertNormal;
    fragVert = vert;
	fragColor = vertColor;
    
    // Apply all matrix transformations to vert
    gl_Position = projector * view * model * vec4(vert, 1);
}
