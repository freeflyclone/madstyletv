#version 330

uniform int mode;
uniform int frameToggle;

precision highp float;

in  vec4 ex_Color;
in  vec2 UV;

// these MUST be set up here with location(layout) else
// they won't be set properly.  glBindFragDataLocation()
// fails if one of these isn't used in the shader code.
layout(location = 0) out vec4 color0;
layout(location = 1) out vec4 color1;
layout(location = 2) out vec4 color2;
layout(location = 3) out vec4 color3;

// these MUST be setup with glUniform1i() prior to main rendering pass
// in order to have access to the FBO pass results
uniform sampler2D texUnit0;
uniform sampler2D texUnit1;
uniform sampler2D texUnit2;
uniform sampler2D texUnit3;

void main(void)
{
    vec4 tc0 = texture(texUnit0, UV);
	vec4 tc1 = texture(texUnit1, UV);
	vec4 tc2 = texture(texUnit2, UV);
	vec4 tc3 = texture(texUnit3, UV);

 	float l0 = tc0.r * 0.21 + tc0.g * 0.72 + tc0.b * 0.07;
 	float l1 = tc1.r * 0.21 + tc1.g * 0.72 + tc1.b * 0.07;
	float diff = abs(abs(l0 - l1)*-1.0);

	if(mode==1) {
		color2 = vec4(diff, diff, diff, 1.0);
		color3 = vec4(1.0, 0.81, 0.0, 1.0);
	}
	else {
		color0 = tc2;
	}
}
