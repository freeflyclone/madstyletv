#include "xgl.h"

XGLMaterial::XGLMaterial() {
	m.ambientColor = white;
	m.diffuseColor = white;
	m.specularColor = white;
	m.shininess = 120.0f;
};

void XGLMaterial::Bind(GLuint program) {
	glUniform4fv(glGetUniformLocation(program, "ambientColor"), 1, glm::value_ptr(m.ambientColor));
	GL_CHECK("glUniform4fv() failed");
	glUniform4fv(glGetUniformLocation(program, "diffuseColor"), 1, glm::value_ptr(m.diffuseColor));
	GL_CHECK("glUniform4fv() failed");
}

