#include "xgl.h"

XGLMaterial::XGLMaterial() {
	a.ambientColor = white;
	a.diffuseColor = white;
	a.specularColor = white;
	a.shininess = 120.0f;
};

void XGLMaterial::Bind(GLuint program) {
	glUniform4fv(l.ambientLocation, 1, glm::value_ptr(a.ambientColor));
	GL_CHECK("glUniform4fv() failed");
	glUniform4fv(l.diffuseLocation, 1, glm::value_ptr(a.diffuseColor));
	GL_CHECK("glUniform4fv() failed");
	glUniform4fv(l.specularLocation, 1, glm::value_ptr(a.specularColor));
	GL_CHECK("glUniform4fv() failed");
	glUniform1f(l.shininessLocation, a.shininess);
	GL_CHECK("glUniform1f() failed");
}

