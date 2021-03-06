#version 150

precision highp float;

uniform vec3 cameraPosition;

layout (std140) uniform LightData {
   vec4 position;
   vec4 intensities; //a.k.a the color of the light
   float attenuation;
   float ambientCoefficient;
} light;

uniform float materialShininess;
uniform vec3 materialSpecularColor;

in  vec4 ex_Color;
in  vec2 UV;

out vec4 out_Color;

void main(void)
{
	float distanceToCurve = -(UV.x * UV.x - UV.y);
	//if (ex_Color.a < 0)
		//distanceToCurve = -distanceToCurve;

	float sdf = smoothstep(-0.001, 0.001, distanceToCurve);

	if (sdf > 0)
		out_Color = ex_Color * sdf;
	else
		discard;
}
