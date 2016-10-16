#version 330

layout (std140) uniform MatrixData {
	mat4 projector;
	mat4 view;
	mat4 model;
};

layout (std140) uniform LightData {
	vec3 pos;
	vec4 color;
};

layout (std140) uniform MaterialData {
    vec4 ambient;
	vec4 diffuse;
	vec4 specular;
    float shininess;
};

uniform sampler2D tex;

in vec3 fragVert;
in vec2 fragTexCoord;
in vec3 fragNormal;
in vec4 fragColor;

out vec4 finalColor;

void main() {
    //calculate normal in world coordinates
    mat3 normalMatrix = transpose(inverse(mat3(model)));
    vec3 normal = normalize(normalMatrix * fragNormal);
    
    //calculate the location of this fragment (pixel) in world coordinates
    vec3 fragPosition = vec3(model * vec4(fragVert, 1));
    
    //calculate the vector from this pixel's surface to the light source
    vec3 surfaceToLight = pos - fragPosition;

    //calculate the cosine of the angle of incidence
    float brightness = dot(normal, surfaceToLight) / (length(surfaceToLight) * length(normal));
    brightness = clamp(brightness, 0, 1);

    //calculate final color of the pixel, based on:
    // 1. The angle of incidence: brightness
    // 2. The color/intensities of the light: light.intensities
    // 3. The texture and texture coord: texture(tex, fragTexCoord)
    //vec4 surfaceColor = texture(tex, fragTexCoord);


    finalColor = (brightness * diffuse * fragColor) + (ambient * 0.1);
    //finalColor = (brightness * diffuse) + (ambient * 0.1);
}
