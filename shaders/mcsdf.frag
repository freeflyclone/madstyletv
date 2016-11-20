#version 150

precision highp float;

in  vec4 ex_Color;
in  vec2 UV;

out vec4 out_Color;

uniform sampler2D texUnit;

float median(float r, float g, float b) {
    return max(min(r, g), min(max(r, g), b));
}

void main(void)
{
	vec4 bgColor = vec4(0.0, 0.0, 0.0, 0.0);
	vec4 fgColor = vec4(1.0, 1.0, 1.0, 1.0);

    vec3 sample = texture(texUnit, UV).rgb;
    float sigDist = median(sample.r, sample.g, sample.b) - 0.5;
    float opacity = clamp(sigDist/fwidth(sigDist) + 0.5, 0.0, 1.0);
    out_Color = mix(bgColor, fgColor, opacity);
}
