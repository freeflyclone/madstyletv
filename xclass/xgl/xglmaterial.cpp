#include "xgl.h"

XGLMaterial::XGLMaterial() {
	attributes.ambientColor = white;
	attributes.diffuseColor = white;
	attributes.specularColor = white;
	attributes.shininess = 120.0f;
};

void XGLMaterial::Bind(GLuint program) {
	glUniform4fv(uniformLocations.ambientLocation, 1, glm::value_ptr(attributes.ambientColor));
	GL_CHECK("glUniform4fv() failed");
	glUniform4fv(uniformLocations.diffuseLocation, 1, glm::value_ptr(attributes.diffuseColor));
	GL_CHECK("glUniform4fv() failed");
	glUniform4fv(uniformLocations.specularLocation, 1, glm::value_ptr(attributes.specularColor));
	GL_CHECK("glUniform4fv() failed");
	glUniform1f(uniformLocations.shininessLocation, attributes.shininess);
	GL_CHECK("glUniform1f() failed");
}

