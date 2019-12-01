#version 150

precision highp float;

in  vec4 ex_Color;
in  vec2 UV;

out vec4 out_Color;

uniform vec4 ambient;
uniform vec4 diffuse;
uniform vec4 specular;
uniform float shininess;
uniform bool monoMode = true;

uniform sampler2D texUnit0;
uniform sampler2D texUnit1;
uniform sampler2D texUnit2;
uniform sampler2D texUnit3;

float radius = 0.05;

float roundedRect(vec2 p, float r) {
	vec2 ulCorner = vec2(r,r);
	vec2 urCorner = vec2(1.0-r, r);
	vec2 lrCorner = vec2(1.0-r, 1.0-r);
	vec2 llCorner = vec2(r, 1.0-r);

	if ((p.x<=r) && (p.y<=r))
		if (distance(ulCorner,p) >= r)
			return 1.0;

	if ((p.x >= (1.0-r)) && (p.y <= r))
		if (distance(urCorner,p) >= r)
			return 1.0;

	if ((p.x >= (1.0-r)) && (p.y >= (1.0-r)))
		if (distance(lrCorner,p) >= r)
			return 1.0;

	if ((p.x <= r) && (p.y >= (1.0-r)))
		if (distance(llCorner,p) >= r)
			return 1.0;

	return 0.0;
}

void main(void)
{
    vec4 tc = texture(texUnit0,UV);

	if(tc.r > 0)
		out_Color = diffuse * vec4(1.0, 1.0, 1.0, tc.r);
	else
		out_Color = ambient;

	float a = roundedRect(UV, radius);

	out_Color = mix(out_Color, vec4(0,0,0,1), a);
}
