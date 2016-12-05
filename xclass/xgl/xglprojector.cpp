#include "xgl.h"

void XGLProjector::Reshape(int w, int h)
{
	width = w;
	height = h;

	Reshape();
}

void XGLProjector::Reshape() {
	glViewport(0, 0, width, height);
	GL_CHECK("glViewport() failed");

	for (auto fn : callbacks)
		fn(width, height);
}

void XGLProjector::AddReshapeCallback(ReshapeFunc fn) {
	callbacks.push_back(fn);
}

glm::mat4 XGLProjector::GetProjectionMatrix() {
	return glm::perspective(glm::radians(45.0f), float(width) / float(height), 0.1f, 1000.0f);
}

glm::mat4 XGLProjector::GetOrthoMatrix() {
	return glm::ortho(0.0f, (float)width, (float)height, 0.0f);
}
