#version 150

precision highp float;

uniform vec4 ambient;
uniform vec4 diffuse;
uniform vec4 specular;
uniform float shininess;

in  vec4 ex_Color;
in  vec2 UV;

out vec4 out_Color;

uniform sampler2D texUnit0;
uniform sampler2D texUnit1;
uniform sampler2D texUnit2;
uniform sampler2D texUnit3;

void main(void)
{
	float y = texture(texUnit0,UV).r;
	float u = texture(texUnit1,UV).r;
	float v = texture(texUnit2,UV).r;

	u = (u - 0.5) * 1.0;
	v = (v - 0.5) * 1.0;

	float r = y + (1.5958 * v);
	float g = y - (0.39173 * u) - (0.81290 * v);
	float b = y + (2.017 * u);

	out_Color = vec4(r,g,b,ambient.a);
}
