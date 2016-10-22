#version 150

precision highp float;

in  vec4 ex_Color;
in  vec2 UV;

out vec4 out_Color;

uniform sampler2D texUnits[8];

void main(void)
{
    //out_Color = ex_Color;
    vec4 tc = texture(texUnits[0],UV);
    out_Color = tc * ex_Color;
}
