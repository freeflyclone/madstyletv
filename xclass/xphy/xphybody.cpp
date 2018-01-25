#include "xphybody.h"

void XPhyBody::SetMatrix() {
	model = glm::translate(glm::mat4(), p) * glm::toMat4(o);
}