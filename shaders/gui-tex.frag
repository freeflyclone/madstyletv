#version 150

precision highp float;

in  vec4 ex_Color;
in  vec2 UV;

out vec4 out_Color;

uniform vec4 ambient;
uniform vec4 diffuse;
uniform vec4 specular;
uniform float shininess;
uniform bool monoMode = true;

uniform sampler2D texUnit0;
uniform sampler2D texUnit1;
uniform sampler2D texUnit2;
uniform sampler2D texUnit3;

void main(void)
{
    vec4 tc0 = texture(texUnit0,UV);

	if (monoMode)
		out_Color = ambient + diffuse * tc0.r;
	else
		out_Color = ambient + diffuse * tc0;
}
