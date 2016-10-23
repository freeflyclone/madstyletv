#version 150

uniform int mode;

precision highp float;

in  vec4 ex_Color;
in  vec2 UV;

out vec4 color0;
out vec4 color1;

uniform sampler2D texUnits[8];

void main(void)
{
    vec4 tc0 = texture(texUnits[0], UV);
	vec4 tc1 = texture(texUnits[1], UV);

 	float l0 = tc0.r * 0.21 + tc0.g * 0.72 + tc0.b * 0.07;

	if(mode==1) {
		color1 = vec4(l0,l0,l0,1.0);
	}
	else {
		color0 = tc1;
	}
}
