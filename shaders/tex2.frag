#version 150

uniform int mode;

precision highp float;

in  vec4 ex_Color;
in  vec2 UV;

out vec4 out_Color;

uniform sampler2D texUnits[8];

void main(void)
{
    vec4 tc = texture(texUnits[0],UV);
 	float l0 = tc.r * 0.21 + tc.g * 0.72 + tc.b * 0.07;
	if(mode==1)
		out_Color = vec4(l0,l0,l0,1.0);
	else
		out_Color = tc * ex_Color;
}
