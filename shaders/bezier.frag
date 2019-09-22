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
uniform vec3 triangle[3];
in  vec4 finalColor;

out vec4 out_Color;

float distance(vec4 p) {
	vec3 p0 = triangle[0];
	vec3 p1 = triangle[1];
	vec3 c = triangle[2];

	return -triangle[0].x * triangle[1].x;
}

void main(void)
{
    out_Color = finalColor * distance(gl_FragCoord);
}
