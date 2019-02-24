/**************************************************************
** Example06BuildScene.cpp
**
** Experiments with XGLSled...
**
** XGLSled is an XGLShape that makes use XPhyBody variables
** to allow for motion simulation. XGLSled uses a Quaternion
** for orientation "local frame" tracking.
**
** This example demonstrates aircraft style control  and +/- thrust 
** methods, connected to a Joystick XInput device
** connects to the XInput joystick interface.
**************************************************************/
#include "ExampleXGL.h"

XGLSled *sled;
XGLShape *cube;

void ExampleXGL::BuildScene() {
	// Add new sled to the scene.  "true" to the constructor turns on display of orientation axes.
	AddShape("shaders/000-simple", [&](){ sled = new XGLSled(true); return sled; });

	// use the left stick to control yaw, right stick to control pitch & roll of the sled (typical R/C transmitter layout)
	// XGLSled::SampleInput(float yaw, float pitch, float roll) also calls XGLSled::GetFinalMatrix()
	AddProportionalFunc("Xbox360Controller0", [this](float v) { sled->SampleInput(-v, 0.0f, 0.0f); });
	AddProportionalFunc("Xbox360Controller2", [this](float v) { sled->SampleInput(0.0f, 0.0f, v); });
	AddProportionalFunc("Xbox360Controller3", [this](float v) { sled->SampleInput(0.0f, -v, 0.0f); });

	// lambda function that moves the sled, to be called by 2 proportional axis callbacks
	XInputProportionalFunc moveFunc = [this](float v) {
		glm::vec4 backward = glm::toMat4(sled->o) * glm::vec4(0.0, v / 10.0f, 0.0, 0.0);

		// update sled's position along it's longitudinal (y) axis, (as though it is an aircraft)
		sled->p += glm::vec3(backward);

		// since we're not using sled->SampleInput(), we need to call 
		// sled->GetFinalMatrix to update sled's position
		sled->model = sled->GetFinalMatrix();

		// make camera track sled
		glm::vec3 lookAtSled = sled->p;
		lookAtSled -= camera.pos;
		camera.Set(camera.pos, lookAtSled, camera.up);
	};

	// move sled with Xbox360 controller left & right triggers
	AddProportionalFunc("Xbox360Controller4", [this, moveFunc](float v) { moveFunc(1.0f + v); });
	AddProportionalFunc("Xbox360Controller5", [this, moveFunc](float v) { moveFunc(-(1.0f + v)); });

	CreateShape("shaders/diffuse", [&]() { cube = new XGLCube(); cube->model = glm::scale(glm::mat4(), glm::vec3(0.2, 1.0, 0.01)); return cube; });
	sled->AddChild(cube);
}
