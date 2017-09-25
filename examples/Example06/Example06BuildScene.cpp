/**************************************************************
** Example06BuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single triangle, with default camera manipulation
** via keyboard and mouse, and joystick movement of the triangle
** around the ground plane with the first 2 axes of the first
** joystick in the system.
**************************************************************/
#include "ExampleXGL.h"

XGLShape *triangle, *cube;

float yaw = 0.0f, pitch = 0.0f;
float x = 10.0f;
float y = 0.0f;
float z = 0.0f;

glm::vec3 pos = { 10.0, 0.0, 0.0 };
glm::vec3 up = { 0.0, 0.0, 1.0 };
glm::vec3 front;

float yawAngle, pitchAngle;

glm::mat4 GetOrientation()
{
	glm::quat q = glm::angleAxis(glm::radians(-pitchAngle), glm::vec3(1, 0, 0));
	q *= glm::angleAxis(glm::radians(-yawAngle), glm::vec3(0, 0, 1));
	return glm::mat4_cast(q);
}

void TwiddleCube() {
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(x, y, z));
	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(1.0, 2.0, 0.01));
	cube->model = translate * GetOrientation() * scale;
}

void ExampleXGL::BuildScene() {
	front = glm::normalize(pos * -1.0f);

	// Simplest of shapes.
	AddShape("shaders/000-simple", [&](){ triangle = new XGLTriangle(); return triangle; });
	AddShape("shaders/diffuse", [&](){ cube = new XGLCube(); return cube; });
	cube->model *= glm::translate(glm::mat4(), glm::vec3(0.0, 10.0, 0.0));

	// if an Xbox 360 controller is attached, use the left joystick X & Y axes to move triangle
	AddProportionalFunc("Xbox360Controller0", [this](float v) { triangle->model[3][0] = v * 10.0f; });
	AddProportionalFunc("Xbox360Controller1", [this](float v) { triangle->model[3][1] = v * 10.0f; });

	// use the right stick to control orientation of the cube
	AddProportionalFunc("Xbox360Controller2", [this](float v) { yaw = v; yawAngle += v; TwiddleCube(); });
	AddProportionalFunc("Xbox360Controller3", [this](float v) { pitch = v; pitchAngle += v; TwiddleCube(); });
}
