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

out vec4 out_Color;

void main(void)
{
    out_Color = ex_Color;
}
