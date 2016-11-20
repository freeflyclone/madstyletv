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
	vec4 color;
	
	color.rgb = ex_Color.rgb;

	if(mask < 0.5)
		color.a = 0.0;
	else
		color.a = 1.0;

	color.a *= smoothstep(0.25, 0.75, mask);

    out_Color = color;
}
