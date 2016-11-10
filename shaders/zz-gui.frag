#version 150

precision highp float;

in  vec4 ex_Color;
in  vec2 UV;

out vec4 out_Color;

uniform vec4 ambient;
uniform vec4 diffuse;
uniform vec4 specular;
uniform float shininess;

uniform sampler2D texUnit0;
uniform sampler2D texUnit1;
uniform sampler2D texUnit2;
uniform sampler2D texUnit3;

void main(void)
{
    vec4 tc = texture(texUnit0,UV);
	vec4 tcMono = vec4(tc.r, tc.r, tc.r, tc.a);
	if (tcMono.r > 0.0)
		tcMono.a = 1.0;
	if( 
		(UV.y <= 3.0/1080.0) || 
		(UV.y >= (1.0 - 3.0/1080.0)) ||
		(UV.x <= 3.0/1920.0) ||
		(UV.x >= (1.0 - 3.0/1920.0))
	)
		out_Color = vec4(1.0, 1.0, 1.0, 0.7);
	else
		out_Color = diffuse * tcMono;
}
