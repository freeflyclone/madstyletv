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

XGLShape *shape;

void ExampleXGL::BuildScene() {
	// Simplest of shapes.
	AddShape("shaders/000-simple", [&](){ shape = new XGLTriangle(); return shape; });

	// if an Xbox 360 controller is attached, use the left joystick X & Y axes to move triangle
	AddProportionalFunc("Xbox360Controller0", [this](float v) { shape->model[3][0] = v * 10.0f; });
	AddProportionalFunc("Xbox360Controller1", [this](float v) { shape->model[3][1] = v * 10.0f; });
}
