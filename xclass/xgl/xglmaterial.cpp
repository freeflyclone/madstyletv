#include "xgl.h"

XGLMaterial::XGLMaterial() {
	attributes.ambientColor = white;
	attributes.diffuseColor = white;
	attributes.specularColor = white;
	attributes.shininess = 120.0f;
};

void XGLMaterial::Bind(GLuint program) {
	glUniform4fv(l.ambientLocation, 1, glm::value_ptr(attributes.ambientColor));
	GL_CHECK("glUniform4fv() failed");
	glUniform4fv(l.diffuseLocation, 1, glm::value_ptr(attributes.diffuseColor));
	GL_CHECK("glUniform4fv() failed");
	glUniform4fv(l.specularLocation, 1, glm::value_ptr(attributes.specularColor));
	GL_CHECK("glUniform4fv() failed");
	glUniform1f(l.shininessLocation, attributes.shininess);
	GL_CHECK("glUniform1f() failed");
}

