#version 330

//uniform mat4 model;
uniform vec3 cameraPosition;

layout (std140) uniform MatrixData {
	mat4 projector;
	mat4 view;
	mat4 model;
};


// material settings
uniform sampler2D materialTex;
uniform float materialShininess;
uniform vec3 materialSpecularColor;

layout (std140) uniform LightData {
   vec4 position;
   vec4 intensities; //a.k.a the color of the light
   float attenuation;
   float ambientCoefficient;
} light;

in vec2 fragTexCoord;
in vec3 fragNormal;
in vec3 fragVert;
in vec3 fragColor;

out vec4 finalColor;

void main() {
    vec3 normal = normalize(transpose(inverse(mat3(model))) * fragNormal);
    vec4 surfacePos = vec4(model * vec4(fragVert, 1));
    //vec4 surfaceColor = texture(materialTex, fragTexCoord);
    vec4 surfaceColor = vec4(fragColor, 1.0);
	vec4 surfaceToLight = normalize(light.position - surfacePos);
    vec3 surfaceToCamera = normalize(cameraPosition - surfacePos.xyz);
    
    //ambient
    vec3 ambient = light.ambientCoefficient * surfaceColor.rgb * light.intensities.rgb;

    //diffuse
    float diffuseCoefficient = max(0.0, dot(normal, surfaceToLight.xyz));
    vec3 diffuse = diffuseCoefficient * surfaceColor.rgb * light.intensities.rgb;
    
    //specular
    float specularCoefficient = 0.0;
    if(diffuseCoefficient > 0.0)
        specularCoefficient = pow(max(0.0, dot(surfaceToCamera, reflect(-surfaceToLight.xyz, normal))), materialShininess);
    vec3 specular = specularCoefficient * materialSpecularColor * light.intensities.rgb;
    
    //attenuation
    float distanceToLight = length(light.position - surfacePos);
    float attenuation = 1.0 / (1.0 + light.attenuation * pow(distanceToLight, 2));

    //linear color (color before gamma correction)
    vec3 linearColor = ambient + attenuation*(diffuse + specular);
    
    //final color (after gamma correction)
    vec3 gamma = vec3(1.0/2.2);
    finalColor = vec4(pow(linearColor, gamma), surfaceColor.a);
}