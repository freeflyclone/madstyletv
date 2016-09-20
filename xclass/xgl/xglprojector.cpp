#include "xgl.h"

void XGLProjector::Reshape(int w, int h)
{
	width = w;
	height = h;

	glViewport(0, 0, width, height);
	GL_CHECK("glViewport() failed");
}

glm::mat4 XGLProjector::GetProjectionMatrix() {
	return glm::perspective(glm::radians(45.0f), float(width) / float(height), 0.1f, 1000.0f);
}

