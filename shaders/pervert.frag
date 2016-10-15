#version 330
 
uniform sampler2D tex;

in Attributes {
    vec4 color;
} AttributesIn;
 
out vec4 outputF;
 
void main() {
 
    outputF = AttributesIn.color;
}