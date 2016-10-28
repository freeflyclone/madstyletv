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

// these MUST be setup with glUniform1i() prior to main rendering pass
// in order to have access to the FBO pass results
uniform sampler2D texUnit0;

void main(void)
{
    vec4 tc0 = texture(texUnit0, UV);
	color0 = tc0;
}
