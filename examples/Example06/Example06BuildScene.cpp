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
	XGLSled() : yaw(0.0f), pitch(0.0f), pos(10.0f, 0.0f, 0.0f), upVector(0.0f, 0.0f, 1.0f), frontVector(0.0f, 1.0f, 0.0f) { 
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
		glDrawArrays(GL_LINES, 0, GLsizei(v.size()));
		GL_CHECK("glDrawArrays() failed");
	}

	glm::mat4 GetOrientation() {
		glm::mat4 orientation;

		rightVector = glm::cross(frontVector, upVector);

		orientation *= glm::rotate(glm::mat4(), glm::radians(-yaw), upVector);
		orientation *= glm::rotate(glm::mat4(), glm::radians(-pitch), rightVector);

		return orientation;
	}

	void Twiddle(float deltaYaw, float deltaPitch) {
		yaw += deltaYaw;
		pitch += deltaPitch;

		glm::mat4 translate = glm::translate(glm::mat4(), pos);

		model = translate * GetOrientation();
	}

private:
	float yaw, pitch;
	XGLVertex pos;
	XGLVertex upVector;
	XGLVertex frontVector;
	XGLVertex rightVector;
};

XGLShape *triangle, *cube;
XGLSled *sled;

void ExampleXGL::BuildScene() {
	// Add new triangle and sled to the scene.
	AddShape("shaders/000-simple", [&](){ triangle = new XGLTriangle(); return triangle; });
	AddShape("shaders/000-simple", [&](){ sled = new XGLSled(); return sled; });

	// Add cube as child of sled, but shape it like a flat rectangle
	CreateShape("shaders/diffuse", [&](){ cube = new XGLCube(); return cube; });
	cube->model = glm::scale(glm::mat4(), glm::vec3(1.0, 2.0, 0.01));
	sled->AddChild(cube);

	// if an Xbox 360 controller is attached, use the left joystick X & Y axes to move triangle
	AddProportionalFunc("Xbox360Controller0", [this](float v) { triangle->model[3][0] = v * 10.0f; });
	AddProportionalFunc("Xbox360Controller1", [this](float v) { triangle->model[3][1] = v * 10.0f; });

	// use the right stick to control orientation of the cube
	AddProportionalFunc("Xbox360Controller2", [this](float v) { sled->Twiddle(v*2.0, 0.0f); });
	AddProportionalFunc("Xbox360Controller3", [this](float v) { sled->Twiddle(0.0f, v*2.0f); });
}
