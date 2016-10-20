#version 150

precision highp float;

in  vec4 ex_Color;
in  vec2 UV;

out vec4 out_Color;

// first shader to utilize multiple texture units
// need a way to communicate the size of the array that's
// determined at runtime.
uniform sampler2D texUnits[8];

void main(void)
{
    vec4 tc0 = texture(texUnits[0],UV);
	vec4 tc1 = texture(texUnits[1],UV);
	
	vec4 tmpColor0 = tc0 * ex_Color;
	vec4 tmpColor1 = tc1 * ex_Color;

	float lumosity0 = tmpColor0.r * 0.21 + tmpColor0.g * 0.72 + tmpColor0.b * 0.07;
	float lumosity1 = tmpColor1.r * 0.21 + tmpColor1.g * 0.72 + tmpColor1.b * 0.07;
	float avg = (lumosity0 + lumosity1) / 2;
	float diff = lumosity1 - lumosity0;

	if(diff>0.1)
		diff = 1;
	else
		diff = 0;


    //out_Color = vec4(lumosity0,lumosity0,lumosity0,1.0);
	out_Color = vec4(avg, avg, avg, 1.0);
	//out_Color = vec4(diff, diff, diff, 1.0);
}
