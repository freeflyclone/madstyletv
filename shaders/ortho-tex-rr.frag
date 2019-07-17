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

float rectangle(vec2 samplePosition, vec2 halfSize)
{
	vec2 componentWiseEdgeDistance = abs(samplePosition) - halfSize;

	return componentWiseEdgeDistance.x;
}

void main(void)
{
    vec4 tc0 = texture(texUnit0,UV);
	vec4 frameColor = vec4(1.0,1.0,1.0,0.8);

	if(tc0.r > 0)
		out_Color = diffuse * vec4(1.0, 1.0, 1.0, tc0.r);
	else
		out_Color = ambient;

	float d = rectangle(UV, vec2(0.03, 1));

	out_Color = mix(frameColor, out_Color, smoothstep(0.298, 0.302, d));
}
