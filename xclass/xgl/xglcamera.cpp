#include "xgl.h"

XGLCamera::XGLCamera() : XGLObject("Camera") {
}

void XGLCamera::Set(glm::vec3 p, glm::vec3 f, glm::vec3 u) {
	pos = p;
	front = f;
	up = u;
}

void XGLCamera::Animate(){
    if (funk)
		funk(this);
}
void XGLCamera::SetTheFunk(XGLCamera::XGLCameraFunk fn){
    funk = fn;
}

glm::mat4 XGLCamera::GetViewMatrix() {
	return glm::lookAt(pos, pos + front, up);
}