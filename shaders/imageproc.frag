#version 150

uniform int mode;
uniform int frameToggle;

precision highp float;

in  vec4 ex_Color;
in  vec2 UV;

// these MUST be set up with glBindFragDataLocation() prior to FBO rendering pass
out vec4 color0;
out vec4 color1;
out vec4 color2;

// these MUST be setup with glUniform1i() prior to main rendering pass
// in order to have access to the FBO pass results
uniform sampler2D texUnit0;
uniform sampler2D texUnit1;
uniform sampler2D texUnit2;

void main(void)
{
    vec4 tc0 = texture(texUnit0, UV);
	vec4 tc1 = texture(texUnit1, UV);
	vec4 tc2 = texture(texUnit2, UV);

 	float l0 = tc0.r * 0.21 + tc0.g * 0.72 + tc0.b * 0.07;

	if(mode==1) {
		color1 = vec4(l0, l0, l0, 1.0);
		color2 = vec4(1.0, 0.81, 0.0, 1.0);
	}
	else {
		color0 = tc1 * tc2;
	}
}
