/**************************************************************
** Example06BuildScene.cpp
**
** Experiments with XGLSled...
**
** XGLSled is an XGLShape with some aircraft-style attitude
** control methods.
**************************************************************/
#include "ExampleXGL.h"

class XGLSled : public XGLShape {
public:
	XGLSled(bool sa = true) : showAxes(sa), position(0.0f, 0.0f, 5.0f) {
		SetName("XGLSled");

		// 3 lines to represent X,Y,Z axes (orientation)
		// X
		v.push_back({ glm::vec3(0), {}, {}, XGLColors::red });
		v.push_back({ glm::vec3(1.0, 0.0, 0.0) * 5.0f, {}, {}, XGLColors::red });
		// Y
		v.push_back({ glm::vec3(0), {}, {}, XGLColors::green });
		v.push_back({ glm::vec3(0.0, 1.0, 0.0) * 5.0f, {}, {}, XGLColors::green });
		// Z
		v.push_back({ glm::vec3(0), {}, {}, XGLColors::blue });
		v.push_back({ glm::vec3(0.0, 0.0, 1.0) * 5.0f, {}, {}, XGLColors::blue });
	}

	void Draw() {
		if (showAxes) {
			glDrawArrays(GL_LINES, 0, 6);
			GL_CHECK("glDrawArrays() failed");
		}
	}

	glm::mat4 GetFinalMatrix() {
		// add the translation of the sled's position for the final model matrix
		return glm::translate(glm::mat4(), position) * glm::toMat4(orientation);
	}

	void SampleInput(float yaw, float pitch, float roll) {
		glm::quat rotation;

		// combine yaw,pitch & roll changes into incremental rotation quaternion
		rotation = glm::angleAxis(glm::radians(yaw), glm::vec3(0.0, 0.0, 1.0));
		rotation *= glm::angleAxis(glm::radians(pitch), glm::vec3(1.0, 0.0, 0.0));
		rotation *= glm::angleAxis(glm::radians(roll), glm::vec3(0.0, 1.0, 0.0));

		// Add combined rotationChange to sled's "currentRotation" (orientation) quaternion
		// This order is key to local-relative rotation or world-relative.  This is local-relative
		// Swapping the operand order changes to world-relative order, which is what I had been doing.
		//
		// Can't believe how long it took to figure this out, because it's SO simple now that I know.
		orientation = orientation * rotation;

		model = GetFinalMatrix();
	}

private:
	bool showAxes;
	XGLVertex position;
	glm::quat orientation;
};

XGLSled *sled;
XGLShape *cube;

void ExampleXGL::BuildScene() {
	// Add new sled to the scene.
	AddShape("shaders/000-simple", [&](){ sled = new XGLSled(); return sled; });

	// use the left stick to control yaw, right stick to control pitch & roll of the sled
	AddProportionalFunc("Xbox360Controller0", [this](float v) { sled->SampleInput(-v, 0.0f, 0.0f); });
	AddProportionalFunc("Xbox360Controller2", [this](float v) { sled->SampleInput(0.0f, 0.0f, v); });
	AddProportionalFunc("Xbox360Controller3", [this](float v) { sled->SampleInput(0.0f, -v, 0.0f); });

	CreateShape("shaders/diffuse", [&]() { cube = new XGLCube(); cube->model = glm::scale(glm::mat4(), glm::vec3(0.2, 1.0, 0.01)); return cube; });
	sled->AddChild(cube);
}
