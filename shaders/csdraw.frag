#version 430

uniform vec4 ambient;
uniform vec4 diffuse;
uniform vec4 specular;
uniform float shininess;

uniform sampler2D srcTex;

in vec2 texCoord;
in vec4 out_Color;

out vec4 color;

void main() {
	vec4 tc = texture(srcTex, texCoord);
	color = tc * diffuse;
}
