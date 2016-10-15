#version 330
 
uniform sampler2D tex;
uniform vec4 ambientColor;
uniform vec4 diffuseColor;

in Attributes {
    vec4 color;
} AttributesIn;
 
out vec4 outputF;
 
void main() {
 
    outputF = AttributesIn.color;
}