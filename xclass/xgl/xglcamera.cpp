#include "xgl.h"

XGLCamera::XGLCamera() : XGLObject("Camera") {
}

void XGLCamera::Set(bool doLookat){
	XGLShaderMatrixData *smd = XGL::getInstance()->GetMatrix();

	if (doLookat) {
		smd->view = glm::lookAt(pos, pos + front, up);
		glBufferSubData(GL_UNIFORM_BUFFER, sizeof(smd->projection), sizeof(smd->view), (GLvoid *)&smd->view);
		GL_CHECK("glBufferSubData() failed.");
	}
}

void XGLCamera::Set(glm::vec3 p, glm::vec3 f, glm::vec3 u) {
	pos = p;
	front = f;
	up = u;

	Set();
}

void XGLCamera::Animate(){
    if (funk)
		funk(this);
}
void XGLCamera::SetTheFunk(XGLCamera::XGLCameraFunk fn){
    funk = fn;
}
