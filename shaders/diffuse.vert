#version 330

uniform mat4 model;

layout (std140) uniform MatrixData {
	mat4 projector;
	mat4 view;
};

layout(location=0) in vec3 vert;
layout(location=1) in vec2 vertTexCoord;
layout(location=2) in vec3 vertNormal;
layout(location=3) in vec4 vertColor;

out vec3 fragVert;
out vec2 fragTexCoord;
out vec3 fragNormal;
out vec4 fragColor;

void main() {
 	// more efficient to do this here instead of fragment shader
    mat3 normalMatrix = transpose(inverse(mat3(model)));

    // Pass some variables to the fragment shader
    fragTexCoord = vertTexCoord;
    fragNormal = normalMatrix * vertNormal;
    fragVert = vert;
	fragColor = vertColor;
    
    // Apply all matrix transformations to vert
    gl_Position = projector * view * model * vec4(vert, 1);
}
