#version 150

precision highp float;

in  vec4 ex_Color;
in  vec2 UV;

out vec4 out_Color;

uniform	mat4 model;

layout (std140) uniform MatrixData {
	mat4 projector;
	mat4 view;
};

uniform vec4 ambient;
uniform vec4 diffuse;
uniform vec4 specular;
uniform float shininess;
uniform bool monoMode = true;

uniform sampler2D texUnit0;
uniform sampler2D texUnit1;
uniform sampler2D texUnit2;
uniform sampler2D texUnit3;

float radius = 0.1;
float fuzz = 0.0015;
vec2 scale = vec2(1.0/model[0][0], 1.0/model[1][1]);

float ellipseFit(vec2 p, vec2 c, float r) {
	vec2 ray = (p - c) * scale;
	return 1.0;
}

float roundedRectangle (vec2 uv, vec2 pos, vec2 size, float radius, float thickness)
{
  float d = length(max(abs(uv - pos), size) - size) - radius;
  return 1.0 - smoothstep(thickness, thickness+0.002, d);
  //return smoothstep(0.66, 0.33, d / thickness * 5.0);
}
void main(void)
{
    vec4 tc = texture(texUnit0,UV);

  vec3 col = vec3(0.0);
  vec2 pos = vec2(0.5, 0.5);
  vec2 size = vec2(0.43, 0.1);
  float radius = 0.06;
  float thickness = 0.0001;

	if(tc.r > 0)
		out_Color = diffuse * vec4(1.0, 1.0, 1.0, tc.r);
	else
		out_Color = ambient;

	float alpha = roundedRectangle (UV, pos, size, radius, thickness);
	 
	out_Color.a *= alpha;
	
	//smoothstep(0.499, 0.501, distance(UV, vec2(0.5,0.5)));
}
