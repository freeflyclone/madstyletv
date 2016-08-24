#include "xgl.h"

void XGLProjector::Reshape(int w, int h)
{
	XGLShaderMatrixData *smd = XGL::getInstance()->GetMatrix();

	width = w;
	height = h;
    glViewport(0, 0, width, height);
	GL_CHECK("glViewport() failed");
	smd->projection = glm::perspective(glm::radians(45.0f), float(width) / float(height), 0.1f, 1000.0f);

	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(smd->projection), (GLvoid *)&smd->view);
	GL_CHECK("glBufferSubData() failed");
}

