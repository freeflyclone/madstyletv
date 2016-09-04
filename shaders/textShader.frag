#version 150

precision highp float;

in  vec4 ex_Color;
in  vec2 UV;

out vec4 out_Color;

uniform sampler2D texUnit;

void main(void)
{
    //out_Color = ex_Color;
    vec2 tc = texture(texUnit,UV).rg;
    out_Color = vec4(1.0, 1.0, 1.0, tc.r) * ex_Color;
}
