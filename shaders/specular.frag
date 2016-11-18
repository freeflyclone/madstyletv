#version 330

uniform vec3 cameraPosition;
uniform mat4 model;

uniform vec4 ambient;
uniform vec4 diffuse;
uniform vec4 specular;
uniform float shininess;

layout (std140) uniform MatrixData {
	mat4 projector;
	mat4 view;
};

layout (std140) uniform LightData {
	vec3 pos;
	vec4 color;
} light;

uniform sampler2D tex;

in vec3 fragVert;
in vec2 fragTexCoord;
in vec3 fragNormal;
in vec4 fragColor;

out vec4 finalColor;

void main() {
    vec3 normal = normalize(fragNormal);

	vec3 surfacePos = vec3(model * vec4(fragVert, 1));
	vec4 surfaceColor = diffuse;
    vec3 surfaceToLight = normalize(light.pos - surfacePos);
    vec3 surfaceToCamera = normalize(cameraPosition - surfacePos);

    vec3 newAmbient = 0.005 * surfaceColor.rgb * light.color.rgb;

	float diffuseCoefficient = max(0.0, dot(normal, surfaceToLight));
	vec3 newDiffuse = diffuseCoefficient * surfaceColor.rgb * light.color.rgb;

    float specularCoefficient = 0.0;
    if(diffuseCoefficient > 0.0)
        specularCoefficient = pow(max(0.0, dot(surfaceToCamera, reflect(-surfaceToLight, normal))), shininess);
    vec3 newSpecular = specularCoefficient * specular.rgb * light.color.rgb;

	float distanceToLight = length(light.pos - surfacePos);
	float attenuation = 1.0 / (1.0 + 0.002 * pow(distanceToLight, 2));

	vec3 linearColor = newAmbient + attenuation*(newDiffuse+newSpecular);

	vec3 gamma = vec3(1.0/2.2);
	
	// cull back facing fragments, to avoid visual turds due to unsorted triangles.
	// does nothing to save from objects overlapping, which is equally unpleasant.
	float alpha = diffuse.a;
	if(!gl_FrontFacing)
		alpha = 0.0;

    finalColor = vec4(pow(linearColor,gamma), alpha);
}
