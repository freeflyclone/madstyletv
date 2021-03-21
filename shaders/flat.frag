#version 330

uniform mat4 model;
uniform vec4 ambient;
uniform vec4 diffuse;
uniform vec4 specular;
uniform float shininess;
uniform sampler2D tex;

layout (std140) uniform MatrixData {
	mat4 projector;
	mat4 view;
};

layout (std140) uniform LightData {
	vec3 pos;
	vec4 color;
};

in vec3 fragVert;
in vec2 fragTexCoord;
in vec3 fragNormal;
in vec4 fragColor;

out vec4 finalColor;

void main() {
    vec3 normal = normalize(fragNormal);
    
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


    //finalColor = (brightness * diffuse * fragColor) + (ambient * 0.1);
    finalColor = diffuse;
}
