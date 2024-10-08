#version 330

precision highp float;

layout (std140) uniform MatrixData {
	mat4 projector;
	mat4 view;
};

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

void main(void)
{
    vec4 tc0 = texture(texUnit0, UV);
	vec4 tc1 = texture(texUnit1, UV);
	color0 = tc1;

	float y = tc0.r * 0.299 + tc0.g * 0.587 + tc0.b * 0.114;
	float u = (tc0.b - y) + 0.5;
	float v = (tc0.r - y) + 0.5;

	color1 = vec4(y,y,y,1.0);
	color2 = vec4(u,u,u,1.0);
	color3 = vec4(v,v,v,1.0);
}
