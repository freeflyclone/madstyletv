#version 150

precision highp float;

in  vec4 ex_Color;
in  vec2 UV;

out vec4 out_Color;

uniform sampler2D texUnit;

void main(void)
{
    vec4 tc = texture(texUnit,UV);
	float mask = tc.a;

	if (mask < 0.5)
		tc.a = 0;
	else
		tc.a = 1.0;

	tc.a *= smoothstep(0.2, 0.8, mask);

    out_Color = tc * ex_Color;
}
