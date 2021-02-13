#version 150

precision highp float;

uniform vec4 ambient;
uniform vec4 diffuse;
uniform vec4 specular;
uniform float shininess;

in  vec4 ex_Color;
in  vec2 UV;

out vec4 out_Color;

uniform sampler2D texUnit0;
uniform sampler2D texUnit1;
uniform sampler2D texUnit2;
uniform sampler2D texUnit3;

void main(void)
{
    float r = texture(texUnit0,UV).r;
    float g = texture(texUnit1,UV).r;
    float b = texture(texUnit2,UV).r;

	vec4 tc = vec4(r,g,b,1.0);
    out_Color = tc;// * ex_Color;
}
