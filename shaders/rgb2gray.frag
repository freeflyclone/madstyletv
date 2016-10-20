#version 150

precision highp float;

in  vec4 ex_Color;
in  vec2 UV;

out vec4 out_Color;

uniform sampler2D texUnit;

void main(void)
{
    //out_Color = ex_Color;
    vec4 tc = texture(texUnit,UV);
	vec4 tmpColor = tc * ex_Color;
	float lumosity = tmpColor.r * 0.21 + tmpColor.g * 0.72 + tmpColor.b * 0.07;
    out_Color = vec4(lumosity,lumosity,lumosity,1.0);
}
